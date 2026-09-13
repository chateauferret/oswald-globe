#ifdef GL_ES
precision mediump float;
#endif

uniform vec3 uWireframeColor;
uniform float uWireframeOpacity;

void main() {
  gl_FragColor = vec4(uWireframeColor, clamp(uWireframeOpacity, 0.0, 1.0));
}
