export type DialogData = {
  name: string,
  url: string,
  probabilities: {[key: string]: number},
  metadata: {[key: string]: any},
}

export type UploadedImage = {
  name: string,
  blob: Blob,
  url: string,
  height?: number,
  width?: number,
  inferenceResponse?: InferenceResponse;
}

export type InferenceResponse = {
  name: string,
  probabilities: {[key: string]: number},
  metadata: {[key: string]: any},
  prediction: Prediction
}

export type Prediction = {
  target: boolean,
  label: number,
  prediction: string
}

export type PublicModels = {
  models: string[]
}

export type SortType =  'ImportanceAsc'
                      | 'ImportanceDesc'
                      | 'NameAsc'
                      | 'NameDesc'
                      | 'None';
