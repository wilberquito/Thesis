<script lang="ts">

  import axios from 'axios';

  import type { UploadedImage } from "$lib/types";
  import Displayer from "./Displayer.svelte";
  import { PUBLIC_URL_SERVICE, PUBLIC_DEFAULT_MODEL } from "$env/static/public";
  import { PUBLIC_MELANOMA_TARGET } from "$env/static/public";

  let uploadedImages: UploadedImage[] = [];

  function onImageClose(i: number) {
    uploadedImages = uploadedImages
      .slice(0, i)
      .concat(uploadedImages.slice(i + 1, uploadedImages.length));
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

  async function fromTaskId(taskId: string) {
    /**
      Uses the taskId to recover the predictions.
      Once the predictions are recovered;

        1) stops the background processes
        2) shows per each images the class that was classified as
    */

    const url = PUBLIC_URL_SERVICE + "/from_task" + `/${taskId}`

    try {
      const resp = await axios.get(url)
      const predictions = resp.data
      console.log(predictions)

      for (const pred of predictions) {
        const target = pred.target
        const imgName = pred.name

        if (target == PUBLIC_MELANOMA_TARGET) {
          const i = uploadedImages.findIndex(e => e.name === imgName)
          if (i >= 0) {
            const img = { ... uploadedImages[i] }
            img.prediction = 'Cancer'
            uploadedImages = uploadedImages
              .slice(0, i)
              .concat(img)
              .concat(uploadedImages.slice(i + 1, uploadedImages.length));
          }
          else {
            console.warn("Trying to update an element that does not exist")
          }
        }
        else {
          const img = uploadedImages.find(e => e.name === imgName)
          if (img) {
            img.prediction = 'NotCancer'
            uploadedImages = [... uploadedImages]
          } else {
            console.warn("Trying to update an element that does not exist")
          }
        }
      }
      console.log(uploadedImages)
    } catch(error)  {
      console.log(error)
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
      const resp = await axios.post(url,
                                    formData, {
                                      params: params,
                                      headers: headers
                                    })
      const taskId = resp.data['task_uuid']
      await fromTaskId(taskId)
    } catch (error) {
      console.error(error);
    }
  }
</script>

<div class="layout">
  <form
    on:submit|preventDefault={() => postImages(uploadedImages)}
    class="file-input-wrapper"
  >
    <label class="btn">
      <p>Select Your Images+</p>
      <input
        type="file"
        class="upload selectable file-input-buttom"
        multiple
        on:change={handleFiles}
      />
    </label>
    <br />
    <label class="upload-btn btn">
      <p>Make prediction</p>
      <input
        type="submit"
        class="upload"
        disabled={uploadedImages.length < 1}
      />
    </label>
  </form>

  {#if uploadedImages.length >= 1}
    <div class="displayer-wrapper">
      <Displayer images={uploadedImages} closeHandler={onImageClose}/>
    </div>
  {/if}
</div>

<style>
  p {
    margin: 0;
  }

  .layout {
    position: relative;
    padding: 1rem 1rem;
    display: flex;
    flex-direction: column;
    max-height: 100%;
  }

   @media only screen and (min-width: 768px) {
    .layout {
      padding: 1rem 10vw;
    }
  }
  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 992px) {
    .layout {
      padding: 1rem 15vw;
    }
  }
  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {
    .layout {
      padding: 1rem 20vw;
    }
  }

  .upload-btn {
    margin-top: 0.5rem;
    width: calc(100% - 14px);
    padding: 1rem 6px !important;
    background-color: #1779ba;
    color: white !important;
  }

  .upload-btn p {
    font-size: 1.2rem !important;
  }

  .upload-btn:has(> input[disabled]) {
    background-color: #cccccc;
    border-color: #4e7691;
    color: #666666;
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
    border: 1px solid #1779ba;
    font-size: 14px;
    color: #1779ba;
  }

  .selectable {
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
    overflow: scroll;
  }

</style>
