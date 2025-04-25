<script lang="ts">
	// Import tailwind styles for the app
	import '../app.css';

	// Import components
	import Header from '$lib/components/Header.svelte';
	import UploadModal from '$lib/components/UploadModal/UploadModal.svelte';
	import { onMount } from 'svelte';

	// Should the upload modal be opened
	let showUploadModal = false;

	// Event handlers

	// Open the modal when the upload button is clicked
	const handleUploadButtonClick = () => {
		showUploadModal = true;
	};

	// Handles keyboard shortcuts
	const handleKeyInput = (e: KeyboardEvent) => {
		if (e.key === 'Escape') showUploadModal = false; // Closes the modal when Escape is pressed
		if (e.key === 'u') showUploadModal = true; // Opens the modal when (U)pload is pressed
	};

	// When the component is first initialised
	onMount(() => {
		document.addEventListener('keydown', handleKeyInput);
		return () => document.removeEventListener('keydown', handleKeyInput);
	});
</script>

<div class="flex h-screen flex-col">
	<Header {handleUploadButtonClick} />

	<main class="m-4 h-full overflow-scroll rounded-lg border border-slate-200 bg-slate-50 shadow-sm">
		<slot />
	</main>

	<UploadModal bind:showModal={showUploadModal} />
</div>
