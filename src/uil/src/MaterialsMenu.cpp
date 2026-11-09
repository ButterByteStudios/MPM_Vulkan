#include <imgui.h>
#include <uil/MaterialsMenu.h>
#include <utility>

#include <cstdint>
#include <string>
#include <functional>
#include <array>
#include <vector>
#include <iostream>

namespace uil
{
    MaterialsMenu::MaterialsMenu(MaterialsManager& manager) : manager(manager)
    { }

    void MaterialsMenu::render(UIStyle& style, ImVec2 position)
    {
        uint32_t uiId = 0;

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(position);

        ImGui::SetNextWindowSize(ImVec2(style.materialsWidth, displaySize.y));

        ImGui::Begin(
            "MaterialsBar",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings
        );

        ImGui::SetCursorPos(ImVec2(style.outerPadding, style.outerPadding));

        if (ImGui::BeginListBox("##00", ImVec2(style.materialsWidth - style.outerPadding * 2, displaySize.y - style.outerPadding * 2)))
        {
            for (int n = 0; n < static_cast<uint32_t>(manager.items.size()); n++)
            {
                ImGui::PushID(uiId);
                uiId++;

                if (manager.items[n].renderInstance(uiId, manager.materialTypes, manager.typeNames))
                {
                    manager.updateMaterialLayout(n);
                }

                ImGui::PopID();
            }

            ImGui::EndListBox();
        }

        if (static_cast<uint32_t>(manager.items.size()) < manager.maxMaterialCount && ImGui::Button("Add"))
        {
            manager.createMaterial(0);
        }

        //ImGui::SameLine();

        // Removing materials is weird because particles would need to be deleted
        //if (selectedIndex != -1 && static_cast<uint32_t>(items.size()) > 1 && ImGui::Button("Remove"))
        //{
        //	items.erase(items.begin() + selectedIndex);
        //	selectedIndex = -1;
        //}

        ImGui::End();
    }
}