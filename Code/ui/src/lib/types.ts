export type Prediction  = 'Cancer' | 'NotCancer'

export type UploadedImage = {
    name: string,
    blob: Blob,
    url: string,
    prediction?: Prediction,
    description?: string
}
