package com.example.mathtools

import android.content.Context
import android.graphics.Bitmap
import com.google.gson.Gson
import com.google.gson.JsonObject
import ai.onnxruntime.*
import java.io.InputStreamReader
import java.nio.charset.Charset

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
    private lateinit var tokenizerConfig: JsonObject
    private lateinit var generationConfig: JsonObject
    private lateinit var modelConfig: JsonObject
    private lateinit var preprocessorConfig: JsonObject
    private val specialTokenIds = mutableSetOf<Long>()
    private val gson = Gson()

    // Параметры генерации и препроцессинга берутся из json рядом с ONNX-моделями.
    private var maxNewTokens = 256
    private var decoderStartTokenId = 2L
    private var eosTokenId = 2L
    private var padTokenId = 0L
    private var imageWidth = 384
    private var imageHeight = 384
    private var rescaleFactor = 1.0f / 255.0f
    private var imageMean = floatArrayOf(0.5f, 0.5f, 0.5f)
    private var imageStd = floatArrayOf(0.5f, 0.5f, 0.5f)

    init {
        loadModel()
    }


    private fun readJsonAsset(path: String): JsonObject {
        context.assets.open(path).use { stream ->
            return gson.fromJson(InputStreamReader(stream), JsonObject::class.java)
        }
    }

    private fun applyModelConfigs() {
        decoderStartTokenId = getConfigLong("decoder_start_token_id", 2L)
        eosTokenId = getConfigLong("eos_token_id", 2L)
        padTokenId = getConfigLong("pad_token_id", 0L)
        maxNewTokens = getConfigLong("max_new_tokens", getConfigLong("max_length", 256L)).toInt()

        val size = preprocessorConfig.getAsJsonObject("size")
        imageWidth = size?.get("width")?.asInt ?: 384
        imageHeight = size?.get("height")?.asInt ?: 384
        rescaleFactor = preprocessorConfig.get("rescale_factor")?.asFloat ?: (1.0f / 255.0f)
        imageMean = readFloatArray(preprocessorConfig.getAsJsonArray("image_mean"), floatArrayOf(0.5f, 0.5f, 0.5f))
        imageStd = readFloatArray(preprocessorConfig.getAsJsonArray("image_std"), floatArrayOf(0.5f, 0.5f, 0.5f))

        specialTokenIds.clear()
        specialTokenIds.add(decoderStartTokenId)
        specialTokenIds.add(eosTokenId)
        specialTokenIds.add(padTokenId)
        tokenizerConfig.getAsJsonObject("added_tokens_decoder")?.entrySet()?.forEach { (id, token) ->
            if (token.asJsonObject.get("special")?.asBoolean == true) {
                specialTokenIds.add(id.toLong())
            }
        }
    }

    private fun getConfigLong(name: String, defaultValue: Long): Long {
        generationConfig.get(name)?.let { if (!it.isJsonNull) return it.asLong }
        modelConfig.get(name)?.let { if (!it.isJsonNull) return it.asLong }
        modelConfig.getAsJsonObject("decoder")?.get(name)?.let { if (!it.isJsonNull) return it.asLong }
        return defaultValue
    }

    private fun readFloatArray(array: com.google.gson.JsonArray?, defaultValue: FloatArray): FloatArray {
        if (array == null || array.size() < 3) return defaultValue
        return FloatArray(3) { index -> array[index].asFloat }
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

            // Загружаем конфиги HuggingFace/Optimum, которые использует оригинальный Pix2Text.
            tokenizer = readJsonAsset("pix2text/tokenizer.json")
            tokenizerConfig = readJsonAsset("pix2text/tokenizer_config.json")
            generationConfig = readJsonAsset("pix2text/generation_config.json")
            modelConfig = readJsonAsset("pix2text/config.json")
            preprocessorConfig = readJsonAsset("pix2text/preprocessor_config.json")
            applyModelConfigs()
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
        // Повторяем параметры TrOCRProcessor/DeiTImageProcessor из preprocessor_config.json:
        // resize -> rescale -> normalize -> NCHW.
        val resized = Bitmap.createScaledBitmap(bitmap, imageWidth, imageHeight, true)

        val pixels = IntArray(imageWidth * imageHeight)
        resized.getPixels(pixels, 0, imageWidth, 0, 0, imageWidth, imageHeight)

        val channels = Array(3) { Array(imageHeight) { FloatArray(imageWidth) } }
        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                val pixel = pixels[y * imageWidth + x]
                val rgb = floatArrayOf(
                    ((pixel shr 16) and 0xFF) * rescaleFactor,
                    ((pixel shr 8) and 0xFF) * rescaleFactor,
                    (pixel and 0xFF) * rescaleFactor
                )

                channels[0][y][x] = (rgb[0] - imageMean[0]) / imageStd[0]
                channels[1][y][x] = (rgb[1] - imageMean[1]) / imageStd[1]
                channels[2][y][x] = (rgb[2] - imageMean[2]) / imageStd[2]
            }
        }

        return arrayOf(channels)
    }

    private fun generateDecoder(encoderState: OnnxTensor): DecoderResult {
        val generated = mutableListOf(decoderStartTokenId.toInt())
        val logitTrace = mutableListOf<String>()

        for (i in 0 until maxNewTokens) {
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
            if (maxIdx.toLong() == eosTokenId) break
        }

        return DecoderResult(
            tokens = generated.map { it.toLong() },
            logitTrace = logitTrace
        )
    }

    private fun decodeTokens(tokens: List<Long>): String {
        val pieces = tokens
            .filterNot { specialTokenIds.contains(it) }
            .mapNotNull { getTokenText(it) }
            .joinToString(separator = "")

        return postProcessLatex(byteLevelDecode(pieces))
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


    private fun byteLevelDecode(text: String): String {
        val byteDecoder = byteLevelDecoder()
        val bytes = ArrayList<Byte>()
        text.forEach { char ->
            val value = byteDecoder[char] ?: char.code
            bytes.add(value.toByte())
        }
        return bytes.toByteArray().toString(Charset.forName("UTF-8"))
    }

    private fun byteLevelDecoder(): Map<Char, Int> {
        val byteToChar = mutableMapOf<Int, Char>()
        val visibleBytes = mutableListOf<Int>()
        visibleBytes.addAll('!'.code..'~'.code)
        visibleBytes.addAll('¡'.code..'¬'.code)
        visibleBytes.addAll('®'.code..'ÿ'.code)

        visibleBytes.forEach { byteToChar[it] = it.toChar() }
        var extra = 0
        for (byte in 0..255) {
            if (!byteToChar.containsKey(byte)) {
                byteToChar[byte] = (256 + extra).toChar()
                extra++
            }
        }
        return byteToChar.entries.associate { (byte, char) -> char to byte }
    }

    private fun postProcessLatex(text: String): String {
        return text
            .replace(Regex("""\\(hat|bar|vec|tilde|dot|ddot)\s*\{\s*}"""), "")
            .replace(Regex("""[\^_]\s*\{\s*}"""), "")
            .replace(Regex("""\\text\s*\{\s*}"""), "")
            .replace(Regex("""\s+"""), " ")
            .replace(" \\", "\\")
            .replace("{ ", "{")
            .replace(" }", "}")
            .trim()
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
