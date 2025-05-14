#include "device.h"

#include <imgui.h>

namespace ovr::nncache {

void 
DeviceNNCache::ui() {

  static struct {
    float ambient{ .6f };
    float diffuse{ .9f };
    float specular{ .4f };
    float shininess{ 40.f };
    float phi{ 99.53f };
    float theta{ 112.2f };
    float intensity{ 1.f };

    bool edit_lm{ false };
    bool wavefront = true;
    int cachemode = 0;
    float lod_scale = 1.f;
    float lod_threshold = 1.0f;
  } locals;

  if (ImGui::Begin("NN-Cache Panel", NULL)) {

    const char* render_modes = " 0 Default (Cache + Network) \0"
                               " 1 Just Cache \0"
                               " 2 Just Network \0";
    if (ImGui::Combo("Cache Mode", &locals.cachemode, render_modes, IM_ARRAYSIZE(render_modes))) {
      ctls.cachemode = locals.cachemode;
    }

    if (ImGui::Checkbox("Wavefront", &locals.wavefront)) {
      ctls.wavefront = locals.wavefront;
    }

    if (ImGui::SliderFloat("LoD Scale", &locals.lod_scale, 0.0f, 10.f, "%.3f")) {
      ctls.lod_scale = locals.lod_scale;
    }

    if (ImGui::SliderFloat("LoD Threshold", &locals.lod_threshold, 0.1f, 10.f, "%.3f")) {
      ctls.lod_threshold = locals.lod_threshold;
    }

    ImGui::Checkbox("Edit Light & Material", &locals.edit_lm); 
    if (locals.edit_lm) {
      bool updated_mat = false;
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Ambient",   &locals.ambient,   0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Diffuse",   &locals.diffuse,   0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Specular",  &locals.specular,  0.f,   1.f, "%.3f");
      updated_mat = updated_mat || ImGui::SliderFloat("Mat: Shininess", &locals.shininess, 0.f, 100.f, "%.3f");
      bool updated_light = false;
      updated_light = updated_light || ImGui::SliderFloat("Light: Phi",       &locals.phi,       0.f, 360.f, "%.2f");
      updated_light = updated_light || ImGui::SliderFloat("Light: Theta",     &locals.theta,     0.f, 360.f, "%.2f");
      updated_light = updated_light || ImGui::SliderFloat("Light: Intensity", &locals.intensity, 0.f,   4.f, "%.3f");
      if (updated_mat) {
        ctls.ambient   = locals.ambient;
        ctls.diffuse   = locals.diffuse;
        ctls.specular  = locals.specular;
        ctls.shininess = locals.shininess;
      }
      if (updated_light) {
        ctls.phi       = locals.phi;
        ctls.theta     = locals.theta;
        ctls.intensity = locals.intensity;
      }
    }

  }
  ImGui::End();
}

}
