<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { API_BASE_URL } from '$lib/api';

	const partition = $page.params.partition;
	
	interface FileInfo {
		link: string;
	}
	
	let files: FileInfo[] = [];
	let loading = true;
	let error: string | null = null;

	async function fetchFiles() {
		try {
			const response = await fetch(`${API_BASE_URL}/partition/${partition}/`);
			
			if (!response.ok) {
				throw new Error(`Failed to fetch files: ${response.status} ${response.statusText}`);
			}
			
			const data = await response.json();
			files = data.files;
			loading = false;
		} catch (err: unknown) {
			error = err instanceof Error ? err.message : 'An unknown error occurred';
			loading = false;
		}
	}

	onMount(fetchFiles);
</script>

<div class="container mx-auto p-4">
	<a href="/" class="text-sky-500 hover:underline mb-4 inline-block focus:outline-none">← Back to Partitions</a>
	
	<h1 class="text-3xl font-bold mb-6">Partition: {partition}</h1>
	
	{#if loading}
		<div class="flex items-center justify-center h-64">
			<p class="text-slate-500">Loading files...</p>
		</div>
	{:else if error}
		<div class="bg-red-50 text-red-600 p-4 rounded-md">
			<p>Error: {error}</p>
		</div>
	{:else if files.length === 0}
		<div class="flex items-center justify-center bg-slate-50 p-8 rounded-lg h-64">
			<p class="text-slate-500">No files found in this partition</p>
		</div>
	{:else}
		<div class="bg-slate-50 p-6 rounded-lg shadow-sm border border-slate-200">
			<h2 class="text-xl font-semibold mb-4">Files ({files.length})</h2>
			<div class="grid grid-cols-1 gap-3">
				{#each files as file}
					<div class="bg-white border border-slate-200 p-3 rounded hover:bg-slate-50 transition-colors">
						<a href={file.link} class="text-sky-500 hover:underline block focus:outline-none">
							{file.link.split('/').pop()}
						</a>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div> 