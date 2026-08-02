package ai.ondevice.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import ai.ondevice.core.PredictionKind
import ai.ondevice.ui.components.NavDestination
import ai.ondevice.ui.screens.AddModelScreen
import ai.ondevice.ui.screens.AllParametersScreen
import ai.ondevice.ui.screens.ChatScreen
import ai.ondevice.ui.screens.DownloadQueueScreen
import ai.ondevice.ui.screens.ImageScreen
import ai.ondevice.ui.screens.LibraryDetailScreen
import ai.ondevice.ui.screens.LibraryScreen
import ai.ondevice.ui.screens.MaskEditorScreen
import ai.ondevice.ui.screens.ModelDetailScreen
import ai.ondevice.ui.screens.ModelsScreen
import ai.ondevice.ui.screens.PromptInspectorScreen
import ai.ondevice.ui.screens.ResolveResultsScreen
import ai.ondevice.ui.screens.RuntimesScreen
import ai.ondevice.ui.screens.SamplerChainScreen
import ai.ondevice.ui.screens.SettingsScreen
import ai.ondevice.ui.screens.ToolsScreen
import ai.ondevice.ui.screens.VideoScreen
import ai.ondevice.ui.screens.VoiceScreen
import ai.ondevice.ui.theme.NIcons

/** Routes. */
object Routes {
    const val CHAT = "chat"
    const val IMAGE = "image"
    const val VOICE = "voice"
    const val LIBRARY = "library"
    const val MODELS = "models"
    const val SETTINGS = "settings"

    // pushes
    const val ADD_MODEL = "models/add"
    const val RESOLVE_RESULTS = "models/resolve"
    const val MODEL_DETAIL = "models/detail/{modelId}"
    const val DOWNLOADS = "models/downloads"
    const val ALL_PARAMETERS = "params/all?runtime={runtime}"
    const val SAMPLER_CHAIN = "params/samplers"

    /** One parameter screen, told which runtime to render. */
    fun parameters(runtime: String) =
        "params/all?runtime=${android.net.Uri.encode(runtime)}"
    const val PROMPT_INSPECTOR = "chat/prompt"
    const val MASK_EDITOR = "image/mask"

    /**
     * Video, pushed from Image rather than given a tab of its own.
     *
     * It runs on the same runtime, the same checkpoints and the same context —
     * a clip generated after a still does not reload — so it belongs where
     * those models already live. The bar is also being kept at five for
     * Workflow, which is the next thing that wants a tab.
     */
    const val VIDEO = "image/video"

    /** One library item, opened. */
    const val LIBRARY_ITEM = "library/item/{kind}/{id}"

    fun libraryItem(kind: PredictionKind, id: String) =
        "library/item/${kind.name}/${android.net.Uri.encode(id)}"

    const val RUNTIMES = "settings/runtimes"
    const val TOOLS = "settings/tools"

    /** A model id is `owner/repo:quant`, so it carries both a slash and a colon. */
    fun modelDetail(modelId: String) = "models/detail/${android.net.Uri.encode(modelId)}"
}

/**
 * Three things this device makes, then the two that describe it: what it has
 * made, and what it can make things with.
 *
 * Video is deliberately not its own entry. It shares the diffusion runtime, the
 * checkpoints and the loaded context with stills, so the two are modes of one
 * tab, switched in the toolbar. "Visuals" and "Sounds" name the domain rather
 * than one direction through it — each holds two modes, and each screen's
 * heading says which one you are in.
 *
 * The sixth slot is being held for Workflow.
 */
val BottomDestinations = listOf(
    NavDestination("Chat", NIcons.Chat, Routes.CHAT),
    NavDestination("Visuals", NIcons.Image, Routes.IMAGE),
    NavDestination("Sounds", NIcons.Voice, Routes.VOICE),
    NavDestination("Library", NIcons.Library, Routes.LIBRARY),
    NavDestination("Settings", NIcons.Settings, Routes.SETTINGS),
)
// Models is no longer one of these.

@Composable
fun OnDeviceApp(
    modifier: Modifier = Modifier,
    initialDestination: String? = null,
    navController: NavHostController = rememberNavController(),
) {
    val backStack by navController.currentBackStackEntryAsState()
    val currentRoute = backStack?.destination?.route

    NavHost(
        navController = navController,
        startDestination = when (initialDestination) {
            ai.ondevice.MainActivity.DEST_DOWNLOADS -> Routes.DOWNLOADS
            else -> Routes.CHAT
        },
        modifier = modifier,
    ) {
        // — root destinations —
        composable(Routes.CHAT) {
            ChatScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onOpenParameters = {
                    navController.navigate(
                        Routes.parameters(ai.ondevice.engine.RuntimeRegistry.LLAMA),
                    )
                },
                onOpenPromptInspector = { navController.navigate(Routes.PROMPT_INSPECTOR) },
                onOpenModels = { navController.navigate(Routes.MODELS) },
            )
        }
        composable(Routes.IMAGE) {
            ImageScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onOpenMask = { navController.navigate(Routes.MASK_EDITOR) },
                onOpenRuntimes = { navController.navigate(Routes.RUNTIMES) },
                onAddModel = { navController.navigate(Routes.ADD_MODEL) },
                onOpenAdvanced = {
                    navController.navigate(
                        Routes.parameters(ai.ondevice.engine.RuntimeRegistry.STABLE_DIFFUSION),
                    )
                },
                onOpenVideo = { navController.navigate(Routes.VIDEO) },
            )
        }
        composable(Routes.VIDEO) {
            VideoScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onBack = { navController.popBackStack() },
                onAddModel = { navController.navigate(Routes.ADD_MODEL) },
                onOpenAdvanced = {
                    navController.navigate(
                        Routes.parameters(ai.ondevice.engine.RuntimeRegistry.STABLE_DIFFUSION),
                    )
                },
            )
        }
        composable(Routes.VOICE) {
            VoiceScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                // The engine decides which parameter set opens.
                onOpenAdvanced = { runtime ->
                    navController.navigate(
                        Routes.parameters(runtime),
                    )
                },
            )
        }
        composable(Routes.LIBRARY) {
            LibraryScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onOpenItem = { kind, id -> navController.navigate(Routes.libraryItem(kind, id)) },
            )
        }
        composable(Routes.SETTINGS) {
            SettingsScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onOpenModels = { navController.navigate(Routes.MODELS) },
                onOpenRuntimes = { navController.navigate(Routes.RUNTIMES) },
                onOpenTools = { navController.navigate(Routes.TOOLS) },
            )
        }

        // — pushes —
        composable(Routes.MODELS) {
            ModelsScreen(
                onBack = { navController.popBackStack() },
                onAddModel = { navController.navigate(Routes.ADD_MODEL) },
                onOpenModel = { navController.navigate(Routes.modelDetail(it)) },
                onOpenDownloads = { navController.navigate(Routes.DOWNLOADS) },
            )
        }
        composable(Routes.ADD_MODEL) {
            AddModelScreen(
                onBack = { navController.popBackStack() },
                onShowRefusals = { navController.navigate(Routes.RESOLVE_RESULTS) },
                onEnterToken = { navController.navigate(Routes.SETTINGS) },
                onDownloadStarted = {
                    navController.navigate(Routes.DOWNLOADS) {
                        popUpTo(Routes.ADD_MODEL) { inclusive = true }
                    }
                },
            )
        }
        composable(Routes.RESOLVE_RESULTS) {
            ResolveResultsScreen(onBack = { navController.popBackStack() })
        }
        composable(Routes.MODEL_DETAIL) { entry ->
            ModelDetailScreen(
                modelId = entry.arguments?.getString("modelId").orEmpty(),
                onBack = { navController.popBackStack() },
                onOpenParameters = { runtime ->
                    navController.navigate(Routes.parameters(runtime))
                },
            )
        }
        composable(Routes.DOWNLOADS) {
            DownloadQueueScreen(onBack = { navController.popBackStack() })
        }
        composable(
            route = Routes.ALL_PARAMETERS,
            arguments = listOf(
                navArgument("runtime") {
                    type = NavType.StringType
                    defaultValue = ai.ondevice.engine.RuntimeRegistry.LLAMA
                },
            ),
        ) { entry ->
            AllParametersScreen(
                onBack = { navController.popBackStack() },
                onOpenSamplerChain = { navController.navigate(Routes.SAMPLER_CHAIN) },
                initialRuntime = entry.arguments?.getString("runtime")
                    ?: ai.ondevice.engine.RuntimeRegistry.LLAMA,
            )
        }
        composable(Routes.SAMPLER_CHAIN) {
            SamplerChainScreen(onBack = { navController.popBackStack() })
        }
        composable(Routes.PROMPT_INSPECTOR) {
            PromptInspectorScreen(onBack = { navController.popBackStack() })
        }
        composable(Routes.MASK_EDITOR) {
            MaskEditorScreen(
                onCancel = { navController.popBackStack() },
                onDone = { navController.popBackStack() },
            )
        }
        composable(Routes.LIBRARY_ITEM) {
            fun openTab(route: String) {
                navController.popBackStack()
                navController.navigateToRoot(route)
            }
            LibraryDetailScreen(
                onBack = { navController.popBackStack() },
                onOpenChat = { openTab(Routes.CHAT) },
                onOpenImage = { openTab(Routes.IMAGE) },
                onOpenVoice = { openTab(Routes.VOICE) },
            )
        }
        composable(Routes.RUNTIMES) {
            RuntimesScreen(onBack = { navController.popBackStack() })
        }
        composable(Routes.TOOLS) {
            ToolsScreen(onBack = { navController.popBackStack() })
        }
    }
}

/** Root destinations are singletons: switching tabs restores rather than stacks, so the bottom bar never accumulates a back stack of its own. */
private fun NavHostController.navigateToRoot(route: String) {
    navigate(route) {
        popUpTo(graph.startDestinationId) { saveState = true }
        launchSingleTop = true
        restoreState = true
    }
}
