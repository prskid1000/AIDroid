package ai.ondevice.engine

/** What an engine says about its own output when that output is wrong. */

fun FloatArray.signalSummary(head: Int = 4): String {
    var peak = 0f
    var sum = 0.0
    var nonZero = 0
    var infinite = 0
    var notANumber = 0
    for (value in this) {
        if (value.isNaN()) {
            notANumber++
            continue
        }
        if (value.isInfinite()) {
            infinite++
            continue
        }
        if (value != 0f) nonZero++
        val magnitude = kotlin.math.abs(value)
        if (magnitude > peak) peak = magnitude
        sum += magnitude
    }
    val finite = size - infinite - notANumber
    val mean = if (finite > 0) sum / finite else 0.0
    val start = if (head > 0 && isNotEmpty()) {
        " head=" + take(head).joinToString { format(it) }
    } else {
        ""
    }
    return "n=$size peak=${format(peak)} mean=${format(mean.toFloat())} " +
        "nonZero=$nonZero inf=$infinite nan=$notANumber$start"
}

/** How much variety a block of token ids has, and what dominates it. */
fun LongArray.codeSummary(): String {
    if (isEmpty()) return "n=0"
    val counts = HashMap<Long, Int>(size / 2 + 1)
    var mode = this[0]
    var modeCount = 0
    for (value in this) {
        val next = (counts[value] ?: 0) + 1
        counts[value] = next
        if (next > modeCount) {
            modeCount = next
            mode = value
        }
    }
    val share = modeCount * 100 / size
    return "n=$size distinct=${counts.size} mode=$mode×$modeCount ($share%)"
}

/** Whether a finished picture is a picture. */
fun DiffusionImage.summary(): String {
    if (pixels.isEmpty()) return "empty"
    var sum = 0.0
    var sumSquares = 0.0
    var black = 0
    var white = 0
    for (pixel in pixels) {
        val luma = ((pixel shr 16 and 0xFF) * 299 + (pixel shr 8 and 0xFF) * 587 +
            (pixel and 0xFF) * 114) / 1000.0
        sum += luma
        sumSquares += luma * luma
        if (luma <= 1.0) black++
        if (luma >= 254.0) white++
    }
    val mean = sum / pixels.size
    val spread = kotlin.math.sqrt((sumSquares / pixels.size - mean * mean).coerceAtLeast(0.0))

    // Mean absolute difference between horizontally adjacent pixels. Low for
    // anything rendered, high for static — and it costs one pass.
    var neighbourSum = 0.0
    var comparisons = 0
    for (y in 0 until height) {
        val row = y * width
        for (x in 1 until width) {
            val a = pixels[row + x] and 0xFF
            val b = pixels[row + x - 1] and 0xFF
            neighbourSum += kotlin.math.abs(a - b).toDouble()
            comparisons++
        }
    }
    val neighbour = if (comparisons > 0) neighbourSum / comparisons else 0.0

    return "${width}×$height mean=${format(mean.toFloat())} spread=${format(spread.toFloat())} " +
        "neighbour=${format(neighbour.toFloat())} " +
        "black=${black * 100 / pixels.size}% white=${white * 100 / pixels.size}%"
}

/** Four significant figures, so a log line stays one line. */
private fun format(value: Float): String = when {
    !value.isFinite() -> value.toString()
    value == 0f -> "0"
    kotlin.math.abs(value) >= 0.001f && kotlin.math.abs(value) < 100_000f ->
        "%.4g".format(value)
    else -> "%.3e".format(value)
}
