<script lang="ts">
	// Import types
	import type { Partition } from '$lib/types';

	// Properties
	let {
		showInput = $bindable(),
		partitions,
		selectedPartition = $bindable()
	}: {
		showInput: boolean;
		partitions: Partition[];
		selectedPartition: Partition | undefined;
	} = $props();
	let searchFilter: string = $state('');

	// Event Handlers

	// Select a partition
	const handleSelectPartition = (partition: Partition) => {
		selectedPartition = partition;
	};

	// Update the search filter
	const handleSearchFilterInput = () => {
		const input = document.getElementById('search-filter-input') as HTMLInputElement;
		searchFilter = input.value;
	};

	// Locally sets a selected partition when a new one should be created
	const handleNewPartitionInput = () => {
		const input = document.getElementById('new-partition-input') as HTMLInputElement;
		if (input.value === '') {
			selectedPartition = partitions[0];
		} else {
			selectedPartition = { partition: input.value, created_at: -1 };
		}
	};
</script>

{#if showInput}
	<div
		class="divide-y-1 absolute left-0 top-full z-10 mt-1 flex w-full cursor-pointer flex-col items-start divide-solid divide-slate-200 rounded-md border border-slate-200 bg-white py-1"
	>
		<input
			id="search-filter-input"
			class="w-full px-4 py-2 pt-1 text-sm italic text-slate-400 placeholder:text-slate-400 focus:outline-none"
			type="text"
			placeholder="Search partitions..."
			oninput={handleSearchFilterInput}
		/>
		{#if partitions.filter((p) => p.partition.includes(searchFilter)).length === 0}
			<span class="w-full cursor-text px-4 py-2 text-start text-sm italic text-slate-400">
				No matching partitions found. Check your spelling, or create a new one.
			</span>
		{/if}
		{#each partitions.filter((p) => p.partition.includes(searchFilter)) as partition}
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<span
				class="w-full px-4 py-2 text-start hover:bg-slate-50"
				onclick={() => {
					handleSelectPartition(partition);
				}}
			>
				{partition.partition}
			</span>
		{/each}
		<input
			id="new-partition-input"
			class="text w-full cursor-pointer px-4 py-2 placeholder:text-pink-500 hover:bg-slate-50 focus:cursor-text focus:outline-none focus:placeholder:text-slate-400"
			type="text"
			placeholder="+ Add a new partition"
			oninput={handleNewPartitionInput}
		/>
	</div>
{/if}
