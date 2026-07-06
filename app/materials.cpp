#include <imgui.h>
#include <materials.h>
#include <string>

MaterialNode::MaterialNode(uint32_t id)
{
    name = "material " + std::to_string(id);
    this->id = id;
}

bool MaterialNode::renderNode()
{
    static int item_selected_idx = 0;

    if (ImGui::TreeNode((name + "###00").c_str()))
    {
        strncpy(buf, name.c_str(), sizeof(buf) - 1);
        ImGui::InputText("Name", buf, sizeof(buf));
        name = buf;
        ImGui::InputFloat("##01", &mat.E, 0.01f, 1.0f, "E = %.3f");
        ImGui::SliderFloat("##02", &mat.v, 0.0f, 0.49f, "v = %.3f");
        ImGui::InputFloat("##03", &mat.rho, 0.01f, 1.0f, "rho = %.3f");
        ImGui::TreePop();
        return true;
    }

    return false; // Change to say true when edited?
}
