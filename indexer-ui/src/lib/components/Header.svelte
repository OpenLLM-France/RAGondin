<script lang="ts">
	// Import utilities
	import type { MouseEventHandler } from 'svelte/elements';

	// Import icons
	import Upload from '$lib/icons/Upload.svelte';

	// Properties
	let {
		handleUploadButtonClick
	}: { handleUploadButtonClick: MouseEventHandler<HTMLButtonElement> } = $props();
</script>

<header class="flex items-center justify-between p-4 pb-0">
	<h1 class="border-b-3 border-b-pink-500 text-2xl font-bold">
		RAGondin Partitions & Files Indexer
	</h1>
	<button
		class="flex btn items-center border-none gap-2 rounded-xl bg-pink-500 font-semibold text-white transition-colors hover:bg-pink-600"
		onclick={handleUploadButtonClick}
	>
		<Upload className="stroke-3 size-5" /> Upload File (U)
	</button>
</header>
