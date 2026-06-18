// MainActivity.kt
package com.example.mathtools

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.webkit.WebView
import android.webkit.WebChromeClient
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import org.json.JSONObject
import java.io.InputStream

class MainActivity : ComponentActivity() {

    private lateinit var webView: WebView
    private lateinit var imageView: ImageView
    private lateinit var rawOutputView: TextView
    private lateinit var ocr: Pix2TextOcr
    private var isMathJaxReady = false
    private var pendingLatex: String? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val inputStream: InputStream? = contentResolver.openInputStream(it)
            val bitmap = inputStream?.use { stream ->
                BitmapFactory.decodeStream(stream)
            }
            bitmap?.let { bmp ->
                processImage(bmp)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        imageView = findViewById(R.id.imageView)
        rawOutputView = findViewById(R.id.rawOutputView)
        val selectButton = findViewById<Button>(R.id.selectButton)

        setupWebView()

        // Инициализируем OCR
        Thread {
            try {
                ocr = Pix2TextOcr(this)
                runOnUiThread {
                    Toast.makeText(this, "Модель загружена", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this, "Ошибка загрузки OCR: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        selectButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
    }

    private fun setupWebView() {
        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                // Проверяем готовность MathJax: скрипт может догрузиться уже после onPageFinished.
                pollMathJaxReady()
            }

            override fun onReceivedError(
                view: WebView?,
                errorCode: Int,
                description: String?,
                failingUrl: String?
            ) {
                Toast.makeText(
                    this@MainActivity,
                    "Ошибка загрузки: $description",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        webView.webChromeClient = WebChromeClient()

        webView.settings.javaScriptEnabled = true
        webView.settings.loadWithOverviewMode = true
        webView.settings.useWideViewPort = true
        webView.settings.domStorageEnabled = true
        webView.settings.allowFileAccess = true
        webView.settings.allowFileAccessFromFileURLs = true
        webView.settings.allowUniversalAccessFromFileURLs = true

        // Загружаем локальную HTML-страницу
        webView.loadUrl("file:///android_asset/mathjax_template.html")
    }

    private fun pollMathJaxReady(attempt: Int = 0) {
        webView.evaluateJavascript("window.mathJaxReady === true") { result ->
            if (result == "true") {
                markMathJaxReady()
            } else if (attempt < 50) {
                mainHandler.postDelayed({ pollMathJaxReady(attempt + 1) }, 200)
            }
        }
    }

    private fun markMathJaxReady() {
        if (isMathJaxReady) return
        isMathJaxReady = true
        Toast.makeText(this@MainActivity, "MathJax готов", Toast.LENGTH_SHORT).show()

        // Если есть ожидающий LaTeX
        pendingLatex?.let { latex ->
            pendingLatex = null
            renderLatex(latex)
        }
    }

    private fun renderLatex(latex: String) {
        if (!isMathJaxReady) {
            pendingLatex = latex
            webView.evaluateJavascript("setRawLatex(${JSONObject.quote(latex)})", null)
            return
        }

        webView.evaluateJavascript("renderLatex(${JSONObject.quote(latex)})") { result ->
            if (result != "true") {
                runOnUiThread {
                    Toast.makeText(this, "Ошибка рендеринга", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun processImage(bitmap: Bitmap) {
        imageView.setImageBitmap(bitmap)

        Thread {
            try {
                if (!::ocr.isInitialized) {
                    throw IllegalStateException("OCR ещё не загружен")
                }
                val result = ocr.recognize(bitmap)
                runOnUiThread {
                    rawOutputView.text = result.rawOutput
                    renderLatex(result.latex)
                    Toast.makeText(this, "Распознано!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    rawOutputView.text = "Ошибка инференса:\n${e.message}"
                    Toast.makeText(this, "Ошибка: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::ocr.isInitialized) {
            ocr.close()
        }
    }
}