export type Prediction  = 'Cancer' | 'NotCancer'


export type ModelInformation = {
  field: string,
  value: string
}

export type DialogData = {
  url: string,
  name: string,
  height?: number,
  width?: number,
  prediction: string,
  target: number,
  model: string,
  probs: {[key:string]: number},
  info: ModelInformation[]
}

export type PredResponse = {
  name: string,
  prediction: string,
  target: number,
  probs: {[key:string]: number}
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
  probs: {[key:string]: number}
}
