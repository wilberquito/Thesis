export type Prediction  = 'Cancer' | 'NotCancer'

export type DialogData = {
  url: string,
  name: string,
  height?: number,
  width?: number,
  prediction: string,
  target: number,
  model: string,
  probability: number
}

export type PredResponse = {
  name: string,
  prediction: string,
  target: number,
  probability: number
}

export type UploadedImage = {
  name: string,
  blob: Blob,
  url: string,
  height?: number,
  width?: number,
  meta?: UploadedImageMetadata;
}

export type UploadedImageMetadata = {
  pred: Prediction,
  prediction: string,
  target: number,
  probability: number
}
