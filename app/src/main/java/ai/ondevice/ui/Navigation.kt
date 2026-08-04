package ai.ondevice.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
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
import ai.ondevice.ui.screens.SamplerChainScreen
import ai.ondevice.ui.screens.SettingsScreen
import ai.ondevice.ui.screens.ToolsScreen
import ai.ondevice.ui.screens.VideoScreen
import ai.ondevice.ui.screens.VoiceScreen
import ai.ondevice.ui.theme.NIcons

/** Routes. */
object Routes {
    /**
     * The graph itself, so a screen can scope a view model to something that
     * outlives its own entry.
     *
     * Video needs it. It is a pushed destination and its Stills toggle pops it,
     * which clears the entry and with it the view model holding a running
     * generation — so leaving the tab for a moment threw away the progress, the
     * resource trace and the clip, while sd.cpp carried on denoising in the
     * background with nothing left to report to.
     */
    const val GRAPH = "graph"

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
    const val ALL_PARAMETERS = "params/all?runtime={runtime}&modelId={modelId}"
    const val SAMPLER_CHAIN = "params/samplers"

    /**
     * One parameter screen, told which runtime to render and whose overrides
     * to edit.
     *
     * The model used to be left out, and the screen guessed: the first
     * installed model of the runtime's modality, most recently used first. So
     * with two diffusion models installed, opening Parameters from one model's
     * page edited the other's row whenever that other one had been used more
     * recently — silently, because nothing on the screen named what it was
     * editing. Per-model settings could not be relied on to reach the model
     * they were set from.
     *
     * Null keeps the guess, which is right for the screens that have no
     * particular model in mind.
     */
    fun parameters(runtime: String, modelId: String? = null) =
        "params/all?runtime=${android.net.Uri.encode(runtime)}" +
            "&modelId=${android.net.Uri.encode(modelId.orEmpty())}"
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

    // A notification tapped while the app is already open arrives as a change
    // to this value rather than as a start destination: the activity is
    // singleTask, so the running instance is reused and the NavHost was built
    // with its start route minutes ago. Without this the download notification
    // would open the app and land wherever it was left.
    LaunchedEffect(initialDestination) {
        if (initialDestination == ai.ondevice.MainActivity.DEST_DOWNLOADS &&
            currentRoute != null && currentRoute != Routes.DOWNLOADS
        ) {
            navController.navigate(Routes.DOWNLOADS)
        }
    }

    NavHost(
        navController = navController,
        startDestination = when (initialDestination) {
            ai.ondevice.MainActivity.DEST_DOWNLOADS -> Routes.DOWNLOADS
            else -> Routes.CHAT
        },
        route = Routes.GRAPH,
        modifier = modifier,
    ) {
        // — root destinations —
        composable(Routes.CHAT) {
            ChatScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                // The conversation's own model, so the screen edits what
                // this tab is talking to rather than whatever was used last.
                onOpenParameters = { modelId ->
                    navController.navigate(
                        Routes.parameters(ai.ondevice.engine.RuntimeRegistry.LLAMA, modelId),
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
                onAddModel = { navController.navigate(Routes.ADD_MODEL) },
                onOpenAdvanced = { modelId ->
                    navController.navigate(
                        Routes.parameters(
                            ai.ondevice.engine.RuntimeRegistry.STABLE_DIFFUSION,
                            modelId,
                        ),
                    )
                },
                onOpenVideo = { navController.navigate(Routes.VIDEO) },
            )
        }
        composable(Routes.VIDEO) { entry ->
            // Scoped to the graph, not to this entry.
            //
            // Stills pops Video off the stack, which clears an entry-scoped
            // view model — so stepping across to the image tab while a clip
            // rendered threw away the progress, the trace and the clip itself,
            // and coming back showed an idle screen over a run that was still
            // going. A generation outlives the screen that started it, so the
            // thing holding it has to as well.
            val graph = remember(entry) { navController.getBackStackEntry(Routes.GRAPH) }
            VideoScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                onBack = { navController.popBackStack() },
                onAddModel = { navController.navigate(Routes.ADD_MODEL) },
                onOpenAdvanced = { modelId ->
                    navController.navigate(
                        Routes.parameters(
                            ai.ondevice.engine.RuntimeRegistry.STABLE_DIFFUSION,
                            modelId,
                        ),
                    )
                },
                viewModel = androidx.hilt.navigation.compose.hiltViewModel(graph),
            )
        }
        composable(Routes.VOICE) {
            VoiceScreen(
                currentRoute = currentRoute,
                onNavigate = { navController.navigateToRoot(it) },
                // The engine decides which parameter set opens, and the tab
                // decides whose overrides it edits.
                onOpenAdvanced = { runtime, modelId ->
                    navController.navigate(Routes.parameters(runtime, modelId))
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
            val modelId = entry.arguments?.getString("modelId").orEmpty()
            ModelDetailScreen(
                modelId = modelId,
                onBack = { navController.popBackStack() },
                // This model's parameters, named. Opening them from a model's
                // own page and landing on another model's overrides is the
                // whole of the bug this argument fixes.
                onOpenParameters = { runtime ->
                    navController.navigate(Routes.parameters(runtime, modelId))
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
                navArgument("modelId") {
                    type = NavType.StringType
                    defaultValue = ""
                },
            ),
        ) { entry ->
            AllParametersScreen(
                onBack = { navController.popBackStack() },
                onOpenSamplerChain = { navController.navigate(Routes.SAMPLER_CHAIN) },
                initialRuntime = entry.arguments?.getString("runtime")
                    ?: ai.ondevice.engine.RuntimeRegistry.LLAMA,
                // Blank is "no particular model" — the screens that open this
                // without one in mind still get the old behaviour.
                initialModelId = entry.arguments?.getString("modelId")
                    ?.takeIf { it.isNotBlank() },
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
