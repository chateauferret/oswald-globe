attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aTexCoord;
attribute float aHeight;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uNMatrix;

varying vec2 vTexCoord;
varying vec3 vNormal;
varying vec3 vObjectNormal;
varying float vHeight;

void main() {
  vTexCoord = aTexCoord;
  vHeight = aHeight;
  vObjectNormal = normalize(aNormal);
  vNormal = normalize(uNMatrix * aNormal);
  gl_Position = uPMatrix * (uMVMatrix * vec4(aPosition, 1.0));
}
