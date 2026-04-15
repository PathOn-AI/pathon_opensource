export interface PointCloudData {
  positions: Float32Array;
  colors: Float32Array;
  normals?: Float32Array;
  pointCount: number;
}

interface PLYHeader {
  format: "ascii" | "binary_little_endian" | "binary_big_endian";
  vertexCount: number;
  properties: Array<{
    name: string;
    type: string;
  }>;
  headerLength: number;
}

function parseHeader(text: string): PLYHeader {
  const lines = text.split("\n");
  let format: PLYHeader["format"] = "ascii";
  let vertexCount = 0;
  const properties: PLYHeader["properties"] = [];
  let headerEndIndex = 0;
  let inVertexElement = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    headerEndIndex += lines[i].length + 1;

    if (line.startsWith("format ")) {
      const formatStr = line.split(" ")[1];
      if (formatStr === "ascii") format = "ascii";
      else if (formatStr === "binary_little_endian") format = "binary_little_endian";
      else if (formatStr === "binary_big_endian") format = "binary_big_endian";
    } else if (line.startsWith("element vertex ")) {
      vertexCount = parseInt(line.split(" ")[2], 10);
      inVertexElement = true;
    } else if (line.startsWith("element ") && inVertexElement) {
      inVertexElement = false;
    } else if (line.startsWith("property ") && inVertexElement) {
      const parts = line.split(" ");
      properties.push({
        type: parts[1],
        name: parts[2],
      });
    } else if (line === "end_header") {
      break;
    }
  }

  return {
    format,
    vertexCount,
    properties,
    headerLength: headerEndIndex,
  };
}

function getTypeSize(type: string): number {
  switch (type) {
    case "char":
    case "uchar":
    case "int8":
    case "uint8":
      return 1;
    case "short":
    case "ushort":
    case "int16":
    case "uint16":
      return 2;
    case "int":
    case "uint":
    case "int32":
    case "uint32":
    case "float":
    case "float32":
      return 4;
    case "double":
    case "float64":
      return 8;
    default:
      return 4;
  }
}

function readValue(
  dataView: DataView,
  offset: number,
  type: string,
  littleEndian: boolean
): number {
  switch (type) {
    case "char":
    case "int8":
      return dataView.getInt8(offset);
    case "uchar":
    case "uint8":
      return dataView.getUint8(offset);
    case "short":
    case "int16":
      return dataView.getInt16(offset, littleEndian);
    case "ushort":
    case "uint16":
      return dataView.getUint16(offset, littleEndian);
    case "int":
    case "int32":
      return dataView.getInt32(offset, littleEndian);
    case "uint":
    case "uint32":
      return dataView.getUint32(offset, littleEndian);
    case "float":
    case "float32":
      return dataView.getFloat32(offset, littleEndian);
    case "double":
    case "float64":
      return dataView.getFloat64(offset, littleEndian);
    default:
      return dataView.getFloat32(offset, littleEndian);
  }
}

export async function parsePLY(file: File): Promise<PointCloudData> {
  const arrayBuffer = await file.arrayBuffer();
  const uint8Array = new Uint8Array(arrayBuffer);

  const decoder = new TextDecoder();
  const headerText = decoder.decode(uint8Array.slice(0, Math.min(10000, uint8Array.length)));
  const endHeaderIndex = headerText.indexOf("end_header");
  if (endHeaderIndex === -1) {
    throw new Error("Invalid PLY file: no end_header found");
  }
  const headerEndPos = endHeaderIndex + "end_header".length + 1;

  const header = parseHeader(headerText);
  const { format, vertexCount, properties } = header;

  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);

  const propIndices: Record<string, number> = {};
  properties.forEach((prop, index) => {
    propIndices[prop.name] = index;
  });

  const hasColors = "red" in propIndices && "green" in propIndices && "blue" in propIndices;
  const hasNormals = "nx" in propIndices && "ny" in propIndices && "nz" in propIndices;

  if (format === "ascii") {
    const dataText = decoder.decode(uint8Array.slice(headerEndPos));
    const lines = dataText.trim().split("\n");

    for (let i = 0; i < Math.min(vertexCount, lines.length); i++) {
      const values = lines[i].trim().split(/\s+/).map(parseFloat);

      positions[i * 3] = values[propIndices["x"]] || 0;
      positions[i * 3 + 1] = values[propIndices["y"]] || 0;
      positions[i * 3 + 2] = values[propIndices["z"]] || 0;

      if (hasColors) {
        colors[i * 3] = (values[propIndices["red"]] || 0) / 255;
        colors[i * 3 + 1] = (values[propIndices["green"]] || 0) / 255;
        colors[i * 3 + 2] = (values[propIndices["blue"]] || 0) / 255;
      } else {
        colors[i * 3] = 0.7;
        colors[i * 3 + 1] = 0.7;
        colors[i * 3 + 2] = 0.7;
      }

      if (hasNormals) {
        normals[i * 3] = values[propIndices["nx"]] || 0;
        normals[i * 3 + 1] = values[propIndices["ny"]] || 0;
        normals[i * 3 + 2] = values[propIndices["nz"]] || 0;
      }
    }
  } else {
    const littleEndian = format === "binary_little_endian";
    const dataView = new DataView(arrayBuffer, headerEndPos);

    let vertexSize = 0;
    const propOffsets: Record<string, { offset: number; type: string }> = {};

    for (const prop of properties) {
      propOffsets[prop.name] = { offset: vertexSize, type: prop.type };
      vertexSize += getTypeSize(prop.type);
    }

    for (let i = 0; i < vertexCount; i++) {
      const baseOffset = i * vertexSize;

      positions[i * 3] = readValue(dataView, baseOffset + propOffsets["x"].offset, propOffsets["x"].type, littleEndian);
      positions[i * 3 + 1] = readValue(dataView, baseOffset + propOffsets["y"].offset, propOffsets["y"].type, littleEndian);
      positions[i * 3 + 2] = readValue(dataView, baseOffset + propOffsets["z"].offset, propOffsets["z"].type, littleEndian);

      if (hasColors) {
        colors[i * 3] = readValue(dataView, baseOffset + propOffsets["red"].offset, propOffsets["red"].type, littleEndian) / 255;
        colors[i * 3 + 1] = readValue(dataView, baseOffset + propOffsets["green"].offset, propOffsets["green"].type, littleEndian) / 255;
        colors[i * 3 + 2] = readValue(dataView, baseOffset + propOffsets["blue"].offset, propOffsets["blue"].type, littleEndian) / 255;
      } else {
        colors[i * 3] = 0.7;
        colors[i * 3 + 1] = 0.7;
        colors[i * 3 + 2] = 0.7;
      }

      if (hasNormals) {
        normals[i * 3] = readValue(dataView, baseOffset + propOffsets["nx"].offset, propOffsets["nx"].type, littleEndian);
        normals[i * 3 + 1] = readValue(dataView, baseOffset + propOffsets["ny"].offset, propOffsets["ny"].type, littleEndian);
        normals[i * 3 + 2] = readValue(dataView, baseOffset + propOffsets["nz"].offset, propOffsets["nz"].type, littleEndian);
      }
    }
  }

  return {
    positions,
    colors,
    normals: hasNormals ? normals : undefined,
    pointCount: vertexCount,
  };
}
