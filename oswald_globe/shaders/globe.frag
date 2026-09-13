#ifdef GL_ES
#extension GL_OES_standard_derivatives : enable
precision mediump float;
#endif

varying vec2 vTexCoord;
varying vec3 vNormal;
varying vec3 vObjectNormal;
varying float vHeight;

uniform sampler2D uSampler;
uniform sampler2D uColorLut;
uniform float uLighting;
uniform float uAtmosphere;
uniform float uGraticule;
uniform float uGraticuleStep;
uniform vec3 uGraticuleColor;
uniform float uGraticuleOpacity;
uniform float uPixelSizeDeg;
uniform float uUseVertexColor;
uniform float uVmin;
uniform float uVmax;

void main() {
  vec4 texColor = texture2D(uSampler, vTexCoord);
  
  // Directional light from top-front-right
  vec3 lightDir = normalize(vec3(0.5, 0.6, 1.0));
  float diff = max(dot(vNormal, lightDir), 0.0);
  float light = mix(1.0, 0.45 + 0.55 * diff, uLighting);

  // Atmospheric rim glow for orthographic projection
  float rim = 1.0 - max(vNormal.z, 0.0);
  rim = pow(rim, 3.5) * uAtmosphere;
  vec3 glowColor = vec3(0.35, 0.65, 1.0);

  // Colorize the interpolated height per-fragment (mesh mode) rather than
  // interpolating already-colorized vertex colors, which would blend RGB
  // linearly across a triangle and produce wrong hues near steep cliffs.
  float heightNorm = clamp((vHeight - uVmin) / max(uVmax - uVmin, 0.0001), 0.0, 1.0);
  vec3 meshColor = texture2D(uColorLut, vec2(heightNorm, 0.5)).rgb;
  vec3 surfaceColor = mix(texColor.rgb, meshColor, uUseVertexColor);
  if (uGraticule > 0.5 && uGraticuleStep > 0.0) {
    // Object-space coordinates keep the grid locked to the rotating globe.
    float latDeg = degrees(asin(clamp(vObjectNormal.y, -1.0, 1.0)));
    float lonDeg = degrees(atan(vObjectNormal.x, vObjectNormal.z));

    float latDist = abs(latDeg - floor(latDeg / uGraticuleStep + 0.5) * uGraticuleStep);
    float lonDist = abs(lonDeg - floor(lonDeg / uGraticuleStep + 0.5) * uGraticuleStep);

    #if defined(GL_OES_standard_derivatives) || !defined(GL_ES)
      float dLat = fwidth(latDeg);
      float dLon = fwidth(lonDeg);
    #else
      float dLat = uPixelSizeDeg;
      float dLon = uPixelSizeDeg / max(0.08, cos(latDeg * 0.0174532925));
    #endif

    float latPix = latDist / max(dLat, 0.00001);
    float lonPix = lonDist / max(dLon, 0.00001);

    float lineLat = 1.0 - smoothstep(0.4, 1.3, latPix);
    float lineLon = 1.0 - smoothstep(0.4, 1.3, lonPix);

    // Major lines: Equator (lat = 0) and Prime Meridian (lon = 0)
    float eqPix = abs(latDeg) / max(dLat, 0.00001);
    float pmPix = abs(lonDeg) / max(dLon, 0.00001);
    float lineMajor = max(1.0 - smoothstep(0.5, 1.5, eqPix), 1.0 - smoothstep(0.5, 1.5, pmPix));

    float gridVal = max(lineLat, lineLon);
    float gridAlpha = mix(gridVal * 0.4, max(gridVal * 0.4, lineMajor * 0.75), lineMajor);
    gridAlpha *= clamp(uGraticuleOpacity, 0.0, 1.0);
    vec3 gridColor = mix(uGraticuleColor, vec3(1.0, 0.85, 0.4), lineMajor * 0.35);

    surfaceColor = mix(surfaceColor, gridColor, gridAlpha);
  }

  vec3 finalRgb = surfaceColor * light + glowColor * rim;
  gl_FragColor = vec4(finalRgb, texColor.a);
}
