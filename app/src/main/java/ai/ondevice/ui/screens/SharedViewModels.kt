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

/**
 * A clip outlives the screen that asked for it.
 *
 * Video is a pushed destination and its Stills toggle pops it, so an
 * entry-scoped view model was cleared the moment you looked at the still tab —
 * taking the progress, the resource trace and the clip about to land with it,
 * while sd.cpp carried on denoising with nothing left to report to. Coming back
 * showed an idle screen over a running generation.
 *
 * Chat, Image and Voice were each given this for the same reason and Video was
 * not, which is the whole of the bug: a run that takes minutes cannot be owned
 * by whichever entry happens to be on the stack.
 */
@Composable
fun activityVideoViewModel(): ai.ondevice.ui.vm.VideoViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

private fun Context.findActivity(): ViewModelStoreOwner {
    var context = this
    while (context is ContextWrapper) {
        if (context is Activity && context is ViewModelStoreOwner) return context
        context = context.baseContext
    }
    error("No Activity ViewModelStoreOwner in this context")
}

/**
 * A workflow run outlives the screen that started it, like every other run —
 * and more so, because a graph is several generations end to end.
 */
@Composable
fun activityWorkflowViewModel(): ai.ondevice.ui.vm.WorkflowViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())
