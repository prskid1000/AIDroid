package ai.ondevice.ui.screens

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModelStoreOwner
import ai.ondevice.ui.vm.ChatViewModel
import ai.ondevice.ui.vm.ParamsViewModel

/**
 * The parameter set is one thing, so the screens that edit it share one
 * ViewModel.
 *
 * S8 (all parameters) and S9 (sampler chain) are separate navigation
 * destinations, and a nav-scoped ViewModel would give each its own copy — so a
 * reorder on S9 would be invisible to S8 until the underlying model row round
 * -tripped through Room, which does not happen at all when no model is loaded.
 * Scoping to the Activity keeps them looking at the same values.
 */
@Composable
fun activityParamsViewModel(): ParamsViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

/**
 * The prompt inspector shows the exact string *this conversation* will send, so
 * it has to be the same conversation the chat screen is holding — a nav-scoped
 * instance would start empty and have nothing to render.
 */
@Composable
fun activityChatViewModel(): ChatViewModel =
    hiltViewModel(viewModelStoreOwner = LocalContext.current.findActivity())

private fun Context.findActivity(): ViewModelStoreOwner {
    var context = this
    while (context is ContextWrapper) {
        if (context is Activity && context is ViewModelStoreOwner) return context
        context = context.baseContext
    }
    error("No Activity ViewModelStoreOwner in this context")
}
