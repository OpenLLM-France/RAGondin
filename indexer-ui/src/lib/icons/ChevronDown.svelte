<script lang="ts">
	// Properties
	let { className = 'size-5 stroke-3' }: { className?: string } = $props();
</script>

<svg
	xmlns="http://www.w3.org/2000/svg"
	viewBox="0 0 24 24"
	class={className}
	fill="none"
	stroke="currentColor"
	stroke-linecap="round"
	stroke-linejoin="round"
>
	<polyline points="5 10 12 17 19 10"></polyline>
</svg>
