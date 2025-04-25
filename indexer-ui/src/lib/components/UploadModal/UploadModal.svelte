<script lang="ts">
	// Import utilities
	import { onMount } from 'svelte';

	// Import types
	import type { Partition } from '$lib/types';

	// Import components
	import PartitionInput from '$lib/components/UploadModal/PartitionInput.svelte';
	import Pip from '$lib/components/Pip.svelte';

	// Import icons
	import Upload from '$lib/icons/Upload.svelte';
	import Close from '$lib/icons/Close.svelte';
	import ChevronDown from '$lib/icons/ChevronDown.svelte';

	// Properties
	let { showModal = $bindable() }: { showModal: boolean } = $props();
	let showDropDown: boolean = $state(false);
	let selectedPartition: Partition | undefined = $state();
	let uploadStatus: string = $state('No file selected.');

	// Data (TMP, need to call API)
	const partitions: Partition[] = [
		{ partition: 'test', created_at: 0 },
		{ partition: 'all', created_at: 0 }
	];

	// Event Handlers

	// Closes the upload modal (and the dropdown, so it's not opened when the modal appears again)
	const handleCloseModal = () => {
		showModal = false;
		showDropDown = false;
	};

	// Toggles open and close the dropdown
	const handleToggleDropdown = () => {
		showDropDown = !showDropDown;
	};

	// Locally sets a selected partition when a new one should be created
	const handleNewPartitionInput = () => {
		const input = document.getElementById('partition-btn') as HTMLInputElement;
		if (input.value === '') {
			selectedPartition = partitions[0];
		} else {
			selectedPartition = { partition: input.value, created_at: -1 };
		}
	};

	// Change uplaod button text to reflect the name of the file after selecting it
	const handleFileUpload = () => {
		const input = document.getElementById('file-upload-btn') as HTMLInputElement;
		uploadStatus = input.value.split('\\').pop() || '';
	};

	// Uploads the file to the partition (TODO) then close the modal ?
	const handleUploadButtonClick = () => {
		console.log('upload');
		handleCloseModal();
	};

	// Handles keyboard shortcuts
	const handleKeyInput = (e: KeyboardEvent) => {
		if (e.key === 'Enter' && e.ctrlKey) handleUploadButtonClick(); // Closes the modal when Ctrl+Enter is pressed
	};

	// When the component is first initialised
	onMount(() => {
		selectedPartition = partitions[0];
	});

	// When the components properties change
	$effect(() => {
		if (showModal) {
			document.addEventListener('keydown', handleKeyInput);
		} else {
			document.removeEventListener('keydown', handleKeyInput);
		}
	});
</script>

{#if showModal}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<div
		class="backdrop-blur-xs absolute h-screen w-screen bg-slate-500/20"
		onclick={handleCloseModal}
	></div>
	<div
		class="-translate-1/2 min-w-144 absolute left-1/2 top-1/2 flex w-1/3 flex-col space-y-4 rounded-lg bg-white p-4 shadow-lg"
	>
		<div class="flex justify-between">
			<h1 class="text-xl font-semibold">Upload File</h1>
			<button onclick={handleCloseModal} class="cursor-pointer">
				<Close className="size-4 stroke-3" />
			</button>
		</div>
		<!-- Partition selection -->
		<div class="relative flex flex-col">
			<label
				class="mb-2 flex cursor-pointer items-center space-x-2 font-medium"
				for="partition-btn"
			>
				<span> Partition </span>
				{#if selectedPartition?.created_at === -1}
					<Pip text="New" />
				{/if}
			</label>
			{#if partitions.length !== 0}
				<button
					id="partition-btn"
					class="btn flex items-center justify-between"
					onclick={handleToggleDropdown}
				>
					<span> {selectedPartition?.partition} </span>
					<ChevronDown className="size-4 stroke-2 stroke-slate-500" />
				</button>
				<PartitionInput bind:showInput={showDropDown} {partitions} bind:selectedPartition />
			{:else}
				<input
					id="partition-btn"
					class="text btn w-full placeholder:text-pink-500 focus:cursor-text focus:outline-none focus:placeholder:text-slate-400"
					type="text"
					placeholder="+ Add a new partition"
					oninput={handleNewPartitionInput}
				/>
			{/if}
		</div>

		<!-- File upload -->
		<div class="relative flex flex-col">
			<label class="mb-2 cursor-pointer font-medium" for="file-upload-btn"> Select a file </label>
			<label class="btn border-dashed border-slate-300" for="file-upload-btn">
				{uploadStatus}
			</label>
			<input id="file-upload-btn" type="file" class="hidden" oninput={handleFileUpload} />
		</div>

		<!-- File metadata -->

		<!-- Cancel and Upload buttons -->
		<div class="flex space-x-4 self-end">
			<button
				class="btn flex items-center gap-2 rounded-xl font-medium text-slate-500 transition-colors"
				onclick={handleCloseModal}
			>
				Cancel (Esc)
			</button>
			<button
				class="btn flex items-center gap-2 rounded-xl border-none bg-pink-500 font-semibold text-white transition-colors hover:bg-pink-600"
				onclick={handleUploadButtonClick}
			>
				<Upload className="stroke-3 size-5" /> Upload File (Ctrl + Enter)
			</button>
		</div>
	</div>
{/if}
