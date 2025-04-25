<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchPartitions } from '$lib/api';
	import type { Partition } from '$lib/types';

	let partitions: Partition[] = [];
	let loading: boolean = true;
	let error: string | null = null;

	const formatDate = (timestamp: number): string => {
		return new Date(timestamp * 1000).toLocaleString("fr");
	};

	onMount(async () => {
		try {
			const data = await fetchPartitions();
			partitions = data.partitions;
			loading = false;
		} catch (err: unknown) {
			console.error(err);
			error = err instanceof Error ? err.message : 'An unknown error occurred';
			loading = false;
		}
	});
</script>

<div class="h-full p-6">
	{#if loading}
		<div class="flex h-full items-center justify-center">
			<p class="text-slate-500">Loading partitions...</p>
		</div>
	{:else if error}
		<div class="flex h-full items-center justify-center">
			<p class="text-red-500">Error: {error}</p>
		</div>
	{:else if partitions.length === 0}
		<div class="flex h-full items-center justify-center">
			<p class="text-lg text-slate-500">No partitions found.</p>
		</div>
	{:else}
		<div class="h-full">
			<div class="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-7">
				{#each partitions as partition}
					<div
						class="cursor-pointer rounded-md bg-white shadow-sm transition-all duration-200 hover:shadow-md"
					>
						<a href="/partition/{partition.partition}" class="block p-5 text-inherit no-underline">
							<div class="mb-2 flex items-center justify-center text-pink-500">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 24 24"
									class="h-12 w-12"
									fill="currentColor"
								>
									<path
										d="M4 4C2.89543 4 2 4.89543 2 6V18C2 19.1046 2.89543 20 4 20H20C21.1046 20 22 19.1046 22 18V8C22 6.89543 21.1046 6 20 6H12L10 4H4Z"
									/>
								</svg>
							</div>
							<div class="mb-1 break-words text-center text-lg font-semibold">
								{partition.partition}
							</div>
							<div class="text-center text-xs text-slate-500">
								Created : {formatDate(partition.created_at)}
							</div>
						</a>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>
