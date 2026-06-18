package com.example.mathtools

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.google.gson.Gson
import com.google.gson.JsonObject
import ai.onnxruntime.*
import java.io.InputStreamReader
import kotlin.math.*

class Pix2TextOcr(private val context: Context) {

    private lateinit var encoderSession: OrtSession
    private lateinit var decoderSession: OrtSession
    private lateinit var tokenizer: JsonObject
    private val gson = Gson()

    // Параметры генерации из generation_config.json
    private val maxLength = 256

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val env = OrtEnvironment.getEnvironment()

            // Загружаем ONNX-модели
            val encoderBytes = context.assets.open("pix2text/encoder_model.onnx").readBytes()
            val decoderBytes = context.assets.open("pix2text/decoder_model.onnx").readBytes()

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            sessionOptions.setInterOpNumThreads(4)
            sessionOptions.setIntraOpNumThreads(4)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            encoderSession = env.createSession(encoderBytes, sessionOptions)
            decoderSession = env.createSession(decoderBytes, sessionOptions)

            // Загружаем токенизатор
            val tokenizerStream = context.assets.open("pix2text/tokenizer.json")
            tokenizer = gson.fromJson(InputStreamReader(tokenizerStream), JsonObject::class.java)
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Ошибка загрузки модели: ${e.message}")
        }
    }

    @Throws(Exception::class)
    fun recognize(imageBitmap: Bitmap): String {
        // 1. Препроцессинг изображения
        val pixelValues = preprocessImage(imageBitmap)

        // 2. Encoder
        val encoderInputs = mapOf("pixel_values" to OnnxTensor.createTensor(encoderSession.environment, pixelValues))
        val encoderOutputs = encoderSession.run(encoderInputs)
        val encoderHiddenState = encoderOutputs.get("last_hidden_state").get() as OnnxTensor

        // 3. Декодер (авторегрессивная генерация)
        val resultTokens = generateDecoder(encoderHiddenState)

        // 4. Декодируем токены в текст (LaTeX)
        return decodeTokens(resultTokens)
    }

    private fun preprocessImage(bitmap: Bitmap): Array<FloatArray> {
        // Resize до 384x384 (как в примере)
        val resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true)

        // Нормализация: (x / 255 - 0.5) / 0.5
        val pixels = IntArray(384 * 384)
        resized.getPixels(pixels, 0, 384, 0, 0, 384, 384)

        val normalized = FloatArray(3 * 384 * 384) // RGB каналы
        for (i in pixels.indices) {
            val r = ((pixels[i] shr 16) and 0xFF) / 255.0f
            val g = ((pixels[i] shr 8) and 0xFF) / 255.0f
            val b = (pixels[i] and 0xFF) / 255.0f

            normalized[i] = (r - 0.5f) / 0.5f
            normalized[i + 384 * 384] = (g - 0.5f) / 0.5f
            normalized[i + 2 * 384 * 384] = (b - 0.5f) / 0.5f
        }

        return arrayOf(normalized)
    }

    private fun generateDecoder(encoderState: OnnxTensor): List<Long> {
        val bosToken = getTokenId("<s>") ?: 0
        val eosToken = getTokenId("</s>") ?: 2

        val generated = mutableListOf(bosToken)

        for (i in 0 until maxLength) {
            // Создаем вход для декодера
            val inputIds = generated.map { it.toLong() }.toLongArray()
            val attentionMask = LongArray(inputIds.size) { 1L }

            val decoderInputs = mapOf(
                "input_ids" to OnnxTensor.createTensor(decoderSession.environment, arrayOf(inputIds)),
                "attention_mask" to OnnxTensor.createTensor(decoderSession.environment, arrayOf(attentionMask)),
                "encoder_hidden_states" to encoderState
            )

            val outputs = decoderSession.run(decoderInputs)
            val logits = outputs.get("logits").get() as OnnxTensor

            // Жадный поиск (берём токен с максимальной вероятностью)
            val logitsArray = logits.value as Array<Array<FloatArray>>
            val lastLogits = logitsArray[0][inputIds.size - 1]
            var maxIdx = 0
            var maxVal = -Float.MAX_VALUE
            for (j in lastLogits.indices) {
                if (lastLogits[j] > maxVal) {
                    maxVal = lastLogits[j]
                    maxIdx = j
                }
            }

            generated.add(maxIdx)

            // Если достигли EOS — завершаем
            if (maxIdx == eosToken) break
        }

        return generated.map { it.toLong() }
    }

    private fun decodeTokens(tokens: List<Long>): String {
        // Парсим tokenizer.json
        val vocab = tokenizer.getAsJsonObject("model").getAsJsonObject("vocab")
        val reverseVocab = mutableMapOf<Long, String>()
        vocab.entrySet().forEach { (token, id) ->
            reverseVocab[id.asLong] = token
        }

        // Преобразуем токены в текст
        val result = StringBuilder()
        for (tokenId in tokens) {
            val tokenStr = reverseVocab[tokenId] ?: continue
            // Пропускаем специальные токены
            if (tokenStr.startsWith("<") && tokenStr.endsWith(">")) continue

            // Заменяем ## на пробел (для BPE-токенизаторов)
            var cleanToken = tokenStr.replace("Ġ", " ").replace("</w>", "")
            result.append(cleanToken)
        }

        return result.toString()
    }

    private fun getTokenId(token: String): Int? {
        val vocab = tokenizer.getAsJsonObject("model").getAsJsonObject("vocab")
        return vocab.get(token)?.asInt
    }

    fun close() {
        try {
            encoderSession.close()
            decoderSession.close()
        } catch (e: Exception) {
            // ignore
        }
    }
}