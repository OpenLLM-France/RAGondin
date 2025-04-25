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
	<line x1="19" y1="5" x2="5" y2="19"></line>
	<line x1="5" y1="5" x2="19" y2="19"></line>
</svg>
