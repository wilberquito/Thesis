<script lang="ts">
  export let models: string[] = [];
  export let selectedModel: string | undefined = undefined;
  export let onModelSelected: (m: string) => void;
</script>


<div class="models-list-wrapper">
    {#each models as model}
    <div
      on:click={() => onModelSelected(model)}
      class="model-item"
      class:model-selected={model === selectedModel}>
      <p>{model}</p>
    </div>
    {/each}
</div>


<style>

  .models-list-wrapper {
    flex: 1;
    width: 100%;
    height: 100%;
    overflow: auto;
    padding-right: 0.5rem;
    border-radius: 0.25rem;
  }

  p {
    margin: 0;
    padding: 0;
  }

  .model-item {
    text-align: right;
    cursor: pointer;
    display: flex;
    justify-content: center;
    border-radius: 0.25rem;
  }

  .model-item:hover {
    background: #eeeeee;
  }

  .model-selected {
    background:  #1779ba !important;
    color: white;
  }

</style>
