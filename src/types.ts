export type LayerType = 'input' | 'conv' | 'relu' | 'maxpool';

export interface LayerDef {
  id: string;
  type: LayerType;
  shape: [number, number, number];
  activations_file: string;
  input_layer?: string;
  groups?: number;
  kernel_size?: number;
  stride?: number;
  padding?: number;
  in_channels?: number;
  weights_file?: string;
  weights_shape?: number[];
  bias_file?: string;
}

export interface ModelMetadata {
  image: string;
  layers: LayerDef[];
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
