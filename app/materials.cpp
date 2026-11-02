#include <imgui.h>
#include <materials.h>
#include <string>

MaterialNode::MaterialNode(uint32_t id, float E, float v, float rho)
{
    name = "material " + std::to_string(id);
    this->id = id;

    Material mat{};
    mat.E = E;
    mat.v = v;
    mat.rho = rho;

    this->mat = mat;
}

bool MaterialNode::renderNode()
{
    static int item_selected_idx = 0;
    bool changed = false;

    if (ImGui::TreeNode((name + "###00").c_str()))
    {
        strncpy(buf, name.c_str(), sizeof(buf) - 1);
        ImGui::InputText("Name", buf, sizeof(buf));
        name = buf;

        ImGui::Text("E = ");
        ImGui::SameLine();
        changed |= ImGui::InputFloat("##01", &mat.E, 1.0f, 100.0f);
        ImGui::Text("v = ");
        ImGui::SameLine();
        changed |= ImGui::SliderFloat("##02", &mat.v, 0.01f, 0.49f);
        ImGui::Text("rho = ");
        ImGui::SameLine();
        changed |= ImGui::InputFloat("##03", &mat.rho, 1.0f, 100.0f);
        ImGui::TreePop();
        return true;
    }

    return changed; // Change to say true when edited?
}
