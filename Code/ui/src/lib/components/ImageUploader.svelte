<script lang="ts">

  import axios from 'axios';

  import type {
    UploadedImage,
    UploadedImageMetadata,
    PredResponse,
    DialogData} from "$lib/types";
  import Displayer from "./Displayer.svelte";
  import LoaderLine from "./LoaderLine.svelte";
  import Dialog from "./Dialog.svelte";
  import { PUBLIC_URL_SERVICE, PUBLIC_DEFAULT_MODEL } from "$env/static/public";
  import { PUBLIC_MELANOMA_TARGET } from "$env/static/public";

  let interactiveText = "Predict";
  let uploadedImages: UploadedImage[] = [];
  let runningPrediction = false;
  let toggledInteractiveButton = -1;
  let dialogData: DialogData | undefined = undefined;

  $: disabledInteractiveButton = uploadedImages.length <= 0 || runningPrediction;
  $: disabledUploadButton = toggledInteractiveButton % 2 === 0;

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
      const img: UploadedImage = {
        name: file.name,
        blob: blob,
        url: URL.createObjectURL(blob),
      };
      uploadedImages = [img, ...uploadedImages];
    }

    // Once all images are loaded, it sort the array by image name
    uploadedImages = uploadedImages.sort((a, b) => (a.name > b.name ? 1 : -1));
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
      const resp  = await axios.get<PredResponse[]>(url)
      const predictions: PredResponse[] = resp.data

      if (onSuccess) onSuccess();

      for (const pred of predictions) {

        const i = uploadedImages.findIndex(e => e.name === pred.name)
        if (i >= 0) {
          const meta: UploadedImageMetadata = {
            pred: pred.target === Number(PUBLIC_MELANOMA_TARGET) ? "Cancer" : "NotCancer",
            target: pred.target,
            probabilities: pred.probabilities,
          }
          const inMemoryImg = uploadedImages[i];
          const img = {
            ... inMemoryImg,
            meta
          }
          uploadedImages = uploadedImages
            .slice(0, i)
            .concat(img)
            .concat(uploadedImages.slice(i + 1, uploadedImages.length));
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
    }
  }

  async function postImages(images: UploadedImage[]) {
    /**
      Post the images to the api service to make the predictions,
      if the post was correct. Then it recover the prediction by taskId
    */
      const formData = new FormData();
      const modelId = PUBLIC_DEFAULT_MODEL
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
        console.warn("The prediction request was canceled because none image found to send")
      }

    try {
      runningPrediction = true;
      const resp = await axios.post(url,
                                    formData, {
                                      params: params,
                                      headers: headers
                                    })
      const taskId = resp.data['task_uuid']
      await fromTaskId(taskId, onPredictionSuccess)
    } catch (error) {
      runningPrediction = false;
      console.error(error);
    }
  }

  function onImageClose(i: number) {
    uploadedImages = uploadedImages
      .slice(0, i)
      .concat(uploadedImages.slice(i + 1, uploadedImages.length));
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
    interactiveText = "Prediction failed, reset"
  }

  function onDialogOpen(i: number) {
    /** Expects the index of the uploaded images opened
        to create the dialog data and show the image
    */
    const img = uploadedImages[i];
    const meta = img.meta;

    if (!meta) {
      console.error('Image metadata is not found')
      return;
    }

    const { name, url} = img;
    const { pred, target, probabilities } = meta;

    const data: DialogData = {
      name: name,
      url,
      pred,
      target,
      probabilities
    }

    dialogData = {... data}
  }

  function onDialogClose() {
    dialogData = undefined;
  }

</script>

{#if runningPrediction}
<div class="loader-position">
  <LoaderLine onLoading={true}></LoaderLine>
</div>
{/if}

<div class="layout-wrapper">

  {#if dialogData}
    <Dialog onClose={onDialogClose}></Dialog>
  {/if}

  <div class="layout">
    <form
      on:submit|preventDefault={() => mayPostImageOrReset(uploadedImages)}
      class="file-input-wrapper"
    >
      <label class="btn select-images"
             class:disabled-btn={disabledUploadButton}>
        <p>Select Your Images+</p>
        <input
          type="file"
          class="upload selectable file-input-buttom"
          multiple
          on:change={handleFiles}
          disabled={disabledUploadButton}
        />
      </label>
      <br />
      <label class="upload-btn btn"
             class:disabled-btn={disabledInteractiveButton}>
        <p>{interactiveText}</p>
        <input
          type="submit"
          class="upload"
          disabled={uploadedImages.length < 1}
        />
      </label>
    </form>

    {#if uploadedImages.length >= 1}
      <div class="displayer-wrapper">
        <Displayer images={uploadedImages}
                   letClose={!disabledUploadButton}
                   closeHandler={onImageClose}
                   expandHandler={onDialogOpen}/>
      </div>
    {/if}
  </div>

    </div>

<style>

  .select-images {
    background-color: #eee;
    border: 1px solid #b3b3b3;

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
    margin-bottom: 0.5rem;
    margin-top: 0.5rem;
    width: calc(100% - 12px);
    padding: 1rem 6px !important;
    background-color: #1779ba;
    color: white !important;
  }

  .upload-btn p {
    font-size: 1.2rem !important;
  }


  input[type="file"] {
    position: absolute;
    z-index: -1;
    top: 10px;
    left: 8px;
    font-size: 17px;
    color: #b8b8b8;
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

</style>
