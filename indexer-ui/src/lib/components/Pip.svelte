<script lang="ts">
	let {
		text,
		color = 'pink'
	}: {
		text: string;
		color?: 'pink' | 'red' | 'blue' | 'green';
	} = $props();

	const colors = {
		pink: 'bg-pink-500',
		red: 'bg-red-500',
		blue: 'bg-blue-500',
		green: 'bg-green-500'
	};
</script>

<span class="rounded-full {colors[color]} px-1 py-0.5 text-[0.5rem] font-bold uppercase text-white">
	{text}
</span>
