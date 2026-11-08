package com.flex.elefin.theme

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color

// Shared colors used by the Jellyseerr request screens and the theme-color picker.
// Elefin Glass palette: true black, graphite surfaces, soft white text.

/** Accent picked in Settings; screens that read JetcasterPrimary recompose when it changes. */
var JetcasterPrimaryColorState by mutableStateOf(Color(0xFFF5F5F7))

val JetcasterPrimary: Color get() = JetcasterPrimaryColorState
val JetcasterBackground = Color(0xFF000000)
val JetcasterOnBackground = Color(0xFFF5F5F7)
val JetcasterSurfaceVariant = Color(0xFF2C2C2E)
val JetcasterOnSurface = Color(0xFFF5F5F7)
