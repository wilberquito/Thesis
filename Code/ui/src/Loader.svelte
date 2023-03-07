<script>
  import ky from "ky";

  let images = [];

  function handleFiles(event) {
    // Removes previous images
    images = [];

    const files = event.target.files;

    for (const file of files) {
      const reader = new FileReader();
      reader.onload = (e) => (images = [...images, e.target.result]);
      reader.readAsDataURL(file);
    }
  }

  async function uploadImages(images) {
    try {
      const formData = new FormData();

      // Append each image to the FormData object
      for (const image of images) {
        formData.append("images[]", image);
      }

      // Makes a post request with all images
      const response = await ky.post("https://example.com/upload", {
        body: formData,
      });

      // Handle the response here
      console.log(await response.json());

    } catch (error) {
      console.error(error);
    }
  }

</script>

<form on:submit|preventDefault={() => uploadImages(images)} class="file-input-wrapper">
  <label class="upload-btn">
    <p>Upload images</p>
    <input
      type="file"
      class="upload file-input-buttom"
      multiple
      on:change={handleFiles}
    />
  </label>
  <label class="upload-btn">
    <p>Make prediction</p>
    <input type="submit" class="upload file-input-buttom" />
  </label>
</form>

<div class="container">
  <div class="img-grid">
    {#each images as image}
      <div class="img-container">
        <img src={image} alt="Possible cancer img" />
      </div>
    {/each}
  </div>
</div>

<style>
  .container {
    margin: auto;
  }

  .img-container {
    width: 100%;
    height: 100%;
  }

  /* mak images fill their container*/
  img {
    object-fit: cover;
    width: 100%;
    height: 100%;
  }

  img:hover {
    opacity: 0.9;
    cursor: pointer;
  }

  /* CSS Grid*/
  .img-grid {
    display: grid;
    grid-template-columns: repeat(1, 1fr);
    grid-gap: 5px;
  }

  /* Media Query for changing grid on bigger screens*/

  /* Bigger than Phones(tablet) */
  @media only screen and (min-width: 768px) {
    .img-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 992px) {
    .img-grid {
      grid-template-columns: repeat(3, 1fr);
    }
  }

  /* Bigger than Phones(laptop / desktop) */
  @media only screen and (min-width: 1200px) {
    .img-grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }

  .file-input-wrapper {
    position: relative;
    display: inline-block;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .upload-btn {
    position: relative;
    display: inline-block;
    font-weight: 600;
    color: #fff;
    text-align: center;
    min-width: 116px;
    padding: 5px;
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid;
    background-color: #4045ba;
    border-color: #4045ba;
    border-radius: 10px;
    line-height: 26px;
    font-size: 14px;
  }

  .upload {
    margin: 0;
  }

  .upload-btn:hover {
    /* background-color: #7a7dcb;
    border-color: #7a7dcb; */
    opacity: 0.9;
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
</style>
