export type Prediction  = 'Cancer' | 'NotCancer'

export type DialogData = {
  url: string,
  name: string,
  pred: string,
  target: number,
  model: string,
  probabilities?: [string, number][]
}

export type PredResponse = {
  name: string,
  pred: string,
  target: number,
  probabilities?: [string,number][]
}

export type UploadedImageMetadata = {
  pred: Prediction,
  target: number,
  probabilities?: [string,number][]
}

export type UploadedImage = {
    name: string,
    blob: Blob,
    url: string,
    meta?: UploadedImageMetadata;
}
