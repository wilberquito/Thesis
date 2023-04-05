export type Prediction  = 'Cancer' | 'NotCancer'

export type DialogData = {
  url: string,
  name: string,
  height?: number,
  width?: number,
  pred: string,
  target: number,
  model: string,
  probability: number
}

export type PredResponse = {
  name: string,
  pred: string,
  target: number,
  probability: number
}

export type UploadedImageMetadata = {
  pred: Prediction,
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
