export type LayerType =
  | 'input'
  | 'conv'
  | 'relu'
  | 'maxpool'
  | 'linear'
  | 'adaptive_avg_pool'
  | 'flatten'
  | 'softmax';

export interface LayerDef {
  id: string;
  type: LayerType;
  shape: [number, number, number] | [number];
  activations_file: string;
  input_layer?: string;
  groups?: number;
  kernel_size?: number;
  stride?: number;
  padding?: number;
  in_channels?: number;
  out_features?: number;
  in_features?: number;
  weights_file?: string;
  weights_shape?: number[];
  bias_file?: string;
}

export interface ModelMetadata {
  image: string;
  layers: LayerDef[];
  predictions?: { class_idx: number; class_name: string; probability: number }[];
  categories?: string[];
}

export interface TensorData {
  data: Float32Array;
  shape: number[];
}

export interface LoadedLayer {
  def: LayerDef;
  activations: TensorData;
  weights?: TensorData;
  bias?: TensorData;
}

export interface ModelData {
  metadata: ModelMetadata;
  layers: Map<string, LoadedLayer>;
  imageUrl: string;
}
