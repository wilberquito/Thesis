export type UploadedImage = {
    name: string,
    blob: Blob,
    url: string
}

export type ImagePayload = {
    name: string,
    base64: string | ArrayBuffer
}