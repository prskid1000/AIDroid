package ai.ondevice.ui.screens

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModelStoreOwner
import ai.ondevice.ui.vm.ChatViewModel
import ai.ondevice.ui.vm.ImageViewModel
import ai.ondevice.ui.vm.ParamsViewModel

/** The parameter set is one thing, so the screens that edit it share one ViewModel. */
@Composable
fun activityParamsViewModel(): ParamsViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

@Composable
fun activityChatViewModel(): ChatViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

/** The image form, its mask and the gallery are one workflow. */
@Composable
fun activityImageViewModel(): ImageViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

@Composable
fun activityVoiceViewModel(): ai.ondevice.ui.vm.VoiceViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

private fun Context.findActivity(): ViewModelStoreOwner {
    var context = this
    while (context is ContextWrapper) {
        if (context is Activity && context is ViewModelStoreOwner) return context
        context = context.baseContext
    }
    error("No Activity ViewModelStoreOwner in this context")
}
