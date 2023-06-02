<script lang="ts">
  export let models: string[] = [];
</script>


<div class="models-list-wrapper">
    {#each models as model}
      <p>{model}</p>
    {/each}
</div>


<style>

  .models-list-wrapper {
    flex: 1;
    width: 100%;
    height: 100%;
    overflow: auto;
  }

  p {
    margin: 0;
    padding: 0;
    text-align: right;
  }

</style>
