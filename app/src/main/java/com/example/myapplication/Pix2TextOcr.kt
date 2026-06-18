package com.example.mathtools

import android.content.Context
import android.graphics.Bitmap
import com.google.gson.Gson
import com.google.gson.JsonObject
import ai.onnxruntime.*
import java.io.InputStreamReader

class Pix2TextOcr(private val context: Context) {

    data class RecognitionResult(
        val latex: String,
        val rawOutput: String
    )

    private data class DecoderResult(
        val tokens: List<Long>,
        val logitTrace: List<String>
    )

    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var encoderSession: OrtSession
    private lateinit var decoderSession: OrtSession
    private lateinit var tokenizer: JsonObject
    private val gson = Gson()

    // Параметры генерации из generation_config.json
    private val maxLength = 256
    private val imageSize = 384

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            // Загружаем ONNX-модели
            val encoderBytes = context.assets.open("pix2text/encoder_model.onnx").readBytes()
            val decoderBytes = context.assets.open("pix2text/decoder_model.onnx").readBytes()

            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            sessionOptions.setInterOpNumThreads(4)
            sessionOptions.setIntraOpNumThreads(4)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            encoderSession = ortEnvironment.createSession(encoderBytes, sessionOptions)
            decoderSession = ortEnvironment.createSession(decoderBytes, sessionOptions)

            // Загружаем токенизатор
            val tokenizerStream = context.assets.open("pix2text/tokenizer.json")
            tokenizer = gson.fromJson(InputStreamReader(tokenizerStream), JsonObject::class.java)
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("Ошибка загрузки модели: ${e.message}")
        }
    }

    @Throws(Exception::class)
    fun recognize(imageBitmap: Bitmap): RecognitionResult {
        // 1. Препроцессинг изображения: ONNX encoder ожидает NCHW [1, 3, 384, 384]
        val pixelValues = preprocessImage(imageBitmap)

        // 2. Encoder
        val encoderInputs = mapOf("pixel_values" to OnnxTensor.createTensor(ortEnvironment, pixelValues))
        val encoderOutputs = encoderSession.run(encoderInputs)
        val encoderHiddenState = encoderOutputs.get("last_hidden_state").get() as OnnxTensor

        // 3. Декодер (авторегрессивная генерация)
        val decoderResult = generateDecoder(encoderHiddenState)

        // 4. Декодируем токены в текст (LaTeX) и возвращаем raw-след логитов для диагностики
        val latex = decodeTokens(decoderResult.tokens)
        val rawOutput = buildString {
            appendLine("LaTeX: $latex")
            appendLine("Token IDs: ${decoderResult.tokens.joinToString(", ")}")
            appendLine("Argmax logits by decoder step:")
            decoderResult.logitTrace.forEach { appendLine(it) }
        }
        return RecognitionResult(latex = latex, rawOutput = rawOutput.trim())
    }

    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        // Resize до 384x384 (как в примере)
        val resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)

        // Нормализация: (x / 255 - 0.5) / 0.5
        val pixels = IntArray(imageSize * imageSize)
        resized.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)

        val channels = Array(3) { Array(imageSize) { FloatArray(imageSize) } }
        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val pixel = pixels[y * imageSize + x]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                channels[0][y][x] = (r - 0.5f) / 0.5f
                channels[1][y][x] = (g - 0.5f) / 0.5f
                channels[2][y][x] = (b - 0.5f) / 0.5f
            }
        }

        return arrayOf(channels)
    }

    private fun generateDecoder(encoderState: OnnxTensor): DecoderResult {
        val bosToken = getTokenId("<s>") ?: 0
        val eosToken = getTokenId("</s>") ?: 2

        val generated = mutableListOf(bosToken)
        val logitTrace = mutableListOf<String>()

        for (i in 0 until maxLength) {
            // Создаем вход для декодера
            val inputIds = generated.map { it.toLong() }.toLongArray()
            val decoderInputs = mapOf(
                "input_ids" to OnnxTensor.createTensor(ortEnvironment, arrayOf(inputIds)),
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
            logitTrace.add(
                "step=$i tokenId=$maxIdx token='${getTokenText(maxIdx.toLong()) ?: "?"}' maxLogit=$maxVal"
            )

            // Если достигли EOS — завершаем
            if (maxIdx == eosToken) break
        }

        return DecoderResult(
            tokens = generated.map { it.toLong() },
            logitTrace = logitTrace
        )
    }

    private fun decodeTokens(tokens: List<Long>): String {
        // Преобразуем токены в текст
        val result = StringBuilder()
        for (tokenId in tokens) {
            val tokenStr = getTokenText(tokenId) ?: continue
            // Пропускаем специальные токены
            if (tokenStr.startsWith("<") && tokenStr.endsWith(">")) continue

            // Заменяем ## на пробел (для BPE-токенизаторов)
            val cleanToken = tokenStr.replace("Ġ", " ").replace("</w>", "")
            result.append(cleanToken)
        }

        return result.toString()
    }

    private fun getTokenId(token: String): Int? {
        val vocab = tokenizer.getAsJsonObject("model").getAsJsonObject("vocab")
        return vocab.get(token)?.asInt
    }

    private fun getTokenText(tokenId: Long): String? {
        val vocab = tokenizer.getAsJsonObject("model").getAsJsonObject("vocab")
        vocab.entrySet().forEach { (token, id) ->
            if (id.asLong == tokenId) return token
        }
        return null
    }

    fun close() {
        try {
            if (::encoderSession.isInitialized) encoderSession.close()
            if (::decoderSession.isInitialized) decoderSession.close()
        } catch (e: Exception) {
            // ignore
        }
    }
}
