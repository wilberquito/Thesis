export type PredictionState  = 'Melanoma' | 'NotWorrying'

export type UploadedImage = {
    name: string,
    blob: Blob,
    url: string,
    prediction?: PredictionState
}
