<script lang="ts">
  import { onMount } from 'svelte';
  import axios from 'axios';
  import {Notifications, acts} from '@tadashi/svelte-notification';
  import type {
    UploadedImage,
    InferenceResponse,
    PublicModels,
    DialogData,
    SortType} from "$lib/types";
  import GridDisplayer from "./GridDisplayer.svelte";
  import ModelsList from "./ModelsList.svelte";
  import LoaderLine from "./LoaderLine.svelte";
  import Dialog from "./Dialog.svelte";
  import { PUBLIC_URL_SERVICE, PUBLIC_DEFAULT_MODEL } from "$env/static/public";

  let interactiveText = "Predict";
  let runningPrediction = false;
  let toggledInteractiveButton = -1;
  let dialogData: DialogData | undefined = undefined;
  let toggledShowModels = false;
  let sortType: SortType = 'None';

  let uploadedImages: UploadedImage[] = [];
  let availableModels: string[] = [];
  let selectedModel: string = PUBLIC_DEFAULT_MODEL

  $: disabledInteractiveButton = uploadedImages.length <= 0 || runningPrediction;
  $: disabledUploadButton = toggledInteractiveButton % 2 === 0;
  $: disabledChangeModel = toggledInteractiveButton % 2 === 0;
  $: disabledModelList = toggledInteractiveButton % 2 === 0;
  $: sortableByImportance = toggledInteractiveButton % 2 === 0;
  $: disableSortable = uploadedImages.length <= 0 || runningPrediction;

  // Loads availables models from api
  onMount(async () => {
    try {
      const url = PUBLIC_URL_SERVICE + "/public_models"
      const resp  = await axios.get<PublicModels>(url)
      availableModels = (resp.data.models) || [];

      if (availableModels.length <= 0) {
        const notification = {
          mode: 'danger',
          message: 'None model available to infer'
        }
        addNotification(notification);
        return;
      }
      availableModels.sort();
      const okDefaultModel = availableModels.includes(selectedModel);
      if (!okDefaultModel) {
        const notification = {
          mode: 'warn',
          message: 'Default model does not appear in the set of available models'
        }
        addNotification(notification);
      }
    } catch(error)  {
      const notification = {
        mode: 'danger',
        message: 'Error stablishing connection to the API'
      }
      addNotification(notification);
    }
  });

  // Adds a notification
  function addNotification(notification: any) {
      acts.add(notification)
  }

  async function handleFiles(event: any) {
    // Treat uploaded images
    const files = event.target.files;

    for (const file of files) {
      // Checks if the image is already loaded
      const inMemory =
        uploadedImages.filter((e) => e.name == file.name).length > 0;

      if (inMemory) continue;

      // If the image is not already loaded, adds the image to the collection
      const blob = new Blob([file], { type: file.type });
      const url = URL.createObjectURL(blob);
      const uploadedImage: UploadedImage = {
        blob,
        url,
        name: file.name,
      };
      const img = new Image();
      img.onload = () => {
        const height = img.height;
        const width = img.width;
        uploadedImage.height = height;
        uploadedImage.width = width;
      }
      img.src = url;
      uploadedImages = [uploadedImage, ...uploadedImages];
    }
  }

  async function fromTaskId(taskId: string,
                            onSuccess?: () => void | undefined,
                            onFailure?: () => void | undefined) {
    /**
      Uses the taskId to recover the predictions.
      Once the predictions are recovered;

        1) stops the background processes
        2) shows per each images the class that was classified as
    */

    const url = PUBLIC_URL_SERVICE + "/from_task" + `/${taskId}`

    try {
      const request = await axios.get<InferenceResponse[]>(url)
      const responseData = request.data;

      if (onSuccess) onSuccess();

      for (const response of responseData) {
        const img = uploadedImages.find(e => e.name === response.name)
        if (img) {
          img.inferenceResponse = response;
          uploadedImages = [...uploadedImages];
        }
        else {
          console.warn("Trying to update an element that does not exist")
        }
      }
    } catch(error)  {
      console.log(error)
      if (onFailure) onFailure()
    }
  }

  async function mayPostImageOrReset(images: UploadedImage[]) {
    toggledInteractiveButton = (toggledInteractiveButton + 1) % 2;
    if (toggledInteractiveButton % 2 === 0) {
      postImages(images);
    }
    else {
      uploadedImages = [];
      interactiveText = "Predict";
      sortType = 'None';
    }
  }

  async function postImages(images: UploadedImage[]) {
    /**
      Post the images to the api service to make the predictions,
      if the post was correct. Then it recover the prediction by taskId
    */
      const formData = new FormData();
      const modelId = selectedModel.trim();
      const url =
        images.length == 1
          ? PUBLIC_URL_SERVICE + "/predict"
          : PUBLIC_URL_SERVICE + "/predict_bulk";

      const params = {
        model_id: modelId
      }

      const headers = {
        'Content-Type': 'multipart/form-data'
      }

      if (images.length == 1) {
        const img = images[0]
        formData.append("file", img.blob, img.name)
      }
      else if(images.length >= 1) {
        for (const img of images) formData.append("files", img.blob, img.name);
      }
      else {
        const notification = {
          mode: 'warn',
          message: 'None image uploaded. Infer request canceled'
        }
        addNotification(notification);
      }

    try {
      runningPrediction = true;
      const resp = await axios.post(url,
                                    formData, {
                                      params: params,
                                      headers: headers
                                    })
      const notification = {
        lifetime: 5,
        mode: 'success',
        message: `Inference model: \n${modelId}`
      }
      addNotification(notification);
      const taskId = resp.data['task_uuid']
      await fromTaskId(taskId, onPredictionSuccess)
    } catch (error) {
      onPredictionFailure();
    }
  }

  function onImageClose(imgName: string) {

    const img = uploadedImages.find(e => e.name === imgName);
    if (img) {
      const notification = {
        lifetime: 2,
        mode: 'warn',
        message: `${img.name} - dropped`
      }
      const idx = uploadedImages.findIndex(e => e.name === imgName);
      addNotification(notification);
      uploadedImages = uploadedImages
        .slice(0, idx)
        .concat(uploadedImages.slice(idx + 1, uploadedImages.length));
    }
    else {
      const notification = {
        lifetime: 2,
        mode: 'warn',
        message: `${imgName} - not pressent in the grid`
      }
      addNotification(notification);
    }
  }

  function onPredictionSuccess() {
    /** Function that resets the state
        when the prediction succeded
    */
    runningPrediction = false;
    interactiveText = "Reset"
  }

  function onPredictionFailure() {
    /** Function that resets the state
        when the prediction succeded
    */
    runningPrediction = false;
    interactiveText = 'Reset';
    const notification = {
      mode: 'danger',
      message: `Bulk of images could not be sended correctly to the API`
    }
    addNotification(notification);
  }

  function onDialogOpen(imgName: string) {
    /** Expects the uid of the uploaded images opened
        to create the dialog data and show the image
    */
    const img = uploadedImages.find(e => e.name === imgName);

    if (img?.inferenceResponse) {
      let metadata: {[key: string]: any} = {...img.inferenceResponse};
      const { height, width } = img;
      metadata = { ...metadata, height, width };
      const { target, prediction, label }  = img.inferenceResponse.prediction;
      metadata = { ...metadata, prediction, target, label };
      const probabilities = img.inferenceResponse.probabilities;
      const { name, url } = img;

      const data: DialogData = {
        name: name,
        url: url,
        metadata: metadata,
        probabilities: probabilities
      }
      dialogData = { ... data };
    } else {
      const notification = {
        mode: 'warn',
        message: `The image - ${imgName} - metadata is not found`
      }
      addNotification(notification);
    }
  }

  function onDialogClose() {
    dialogData = undefined;
  }

  function toggleSelectionModel() {
    toggledShowModels = !toggledShowModels;
  }

  function onModelSelected(model: string) {
    selectedModel = model;
  }

  function sortGrid() {

    if (sortableByImportance) {
      if (sortType === 'None') {
        sortType = 'ImportanceAsc';
      } else if (sortType === 'ImportanceAsc') {
        sortType = 'ImportanceDesc';
      } else if (sortType === 'ImportanceDesc') {
        sortType = 'ImportanceAsc';
      } else {
        sortType = 'ImportanceAsc';
      }
    } else {
      if (sortType === 'None') {
        sortType = 'NameAsc';
      } else if (sortType === 'NameAsc') {
        sortType = 'NameDesc';
      } else if (sortType === 'NameDesc') {
        sortType = 'NameAsc';
      } else {
        sortType = 'NameAsc';
      }
    }

    const notification = {
      lifetime: 2,
      mode: 'info',
      message: `Grid sorted by ${sortType}`
    }
    addNotification(notification);
  }

</script>


<Notifications />

{#if runningPrediction}
<div class="loader-position">
  <LoaderLine onLoading={true}></LoaderLine>
</div>
{/if}

<div class="layout-wrapper">

  {#if dialogData}
    <Dialog dialogData={dialogData}
      onClose={onDialogClose}></Dialog>
  {/if}

  <div class="layout">
    <form
      on:submit|preventDefault={() => mayPostImageOrReset(uploadedImages)}
      class="file-input-wrapper"
    >
      <div class="line-wrapper">
        <label class="upload-btn btn"
               class:disabled-btn={disabledInteractiveButton}>
          <p>{interactiveText}</p>
          <input
            type="submit"
            class="upload"
            disabled={uploadedImages.length < 1}
          />
        </label>
      </div>

      <div class="line-wrapper">

      <label class="tool-selection select-images"
             class:disabled-btn={disabledUploadButton}>
        <span class="material-icons">
        image
                    </span>
        <input
          type="file"
          class="upload selectable file-input-buttom"
          multiple
          on:change={handleFiles}
          disabled={disabledUploadButton}
        />
      </label>

      {#if sortableByImportance }
        <button class="tool-selection"
                type="button"
                disabled={disableSortable}
                class:disabled-btn={disableSortable}
                on:click={sortGrid}>
          <span class="material-icons">
          low_priority
          </span>
        </button>
      {:else}
        <button class="tool-selection"
                type="button"
                disabled={disableSortable}
                class:disabled-btn={disableSortable}
                on:click={sortGrid}>
          <span class="material-icons">
          sort_by_alpha
          </span>
        </button>
      {/if}
        <button class="tool-selection"
                type="button"
                disabled={disabledChangeModel}
                class:disabled-btn={disabledUploadButton}
                on:click={toggleSelectionModel}>
          <span class="material-icons">
          hub
          </span>
        </button>

        <div class="model-list-wrapper"
          class:hide-model-list={!toggledShowModels}
          class:disabled-btn={disabledModelList}>
          <ModelsList
            models={availableModels}
            selectedModel={selectedModel}
            onModelSelected={onModelSelected}
          />
        </div>

      </div>

    </form>

    {#if uploadedImages.length >= 1}
      <div class="displayer-wrapper">
        <GridDisplayer images={uploadedImages}
                       letClose={!disabledUploadButton}
                       closeHandler={onImageClose}
                       expandHandler={onDialogOpen}
                       sortType={sortType}/>
      </div>
    {/if}
  </div>

    </div>

<style>

  .hide-model-list {
    display: none;
  }

  .model-list-wrapper {
    flex: 1;
    font-size: 0.8rem;
  }

  .model-list-wrapper.disabled-btn :global(.model-item) {
    pointer-events: none;
  }

  .disabled-btn {
    opacity: 0.7;
    cursor: default !important;
  }

  .disabled-btn input {
    cursor: default !important;
  }

  p {
    margin: 0;
  }

  .layout-wrapper {
    position: relative;
    padding: 1rem 1rem;
    height: calc(100% - 2rem);
  }

  .layout {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }

  @media only screen and (min-width: 768px) {
    .layout-wrapper {
      padding: 1rem 10vw;
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 992px) {
    .layout-wrapper {
      padding: 1rem 15vw;
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {
    .layout-wrapper {
      padding: 1rem 20vw;
    }
  }

  .upload-btn {
    width: calc(100% - 12px);
    padding: 1rem 6px !important;
    background-color: #1779ba;
    color: white !important;
  }

  .upload-btn p {
    font-size: 1.2rem !important;
  }


  input[type="file"] {
    display: none;
  }

  .file-input-wrapper {
    position: relative;
  }

  .btn {
    position: relative;
    border-radius: 0.25rem;
    display: inline-block;
    font-weight: 600;
    text-align: center;
    min-width: 116px;
    padding: 1rem 6px;
    transition: all 0.3s ease;
    font-size: 14px;
    color: #1779ba;
    cursor: pointer;
  }

  .btn p {
    font-size: 0.8rem;
  }

  .file-input-wrapper input {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
  }

  .displayer-wrapper {
    position: relative;
    overflow-y: scroll;
  }

  .loader-position {
    width: 100%;
    position: absolute;
  }

  .line-wrapper {
    display: flex;
    margin-bottom: 0.5rem;
    height: 4rem;
    gap: 0.25rem;
  }

  button {
    all: unset;
  }

  .tool-selection {
    border: 1px solid #b3b3b3;
    background-color: #eee;
    width: 4.5rem;
    height: 100%;
    color: #1779ba;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 0.25rem;
  }

</style>
