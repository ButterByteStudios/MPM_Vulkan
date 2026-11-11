#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <uil/UIManager.h>
#include <uil/UIStyle.h>

namespace uil
{
    UIManager::UIManager(MaterialsManager& materialsManager) : materialsMenu(MaterialsMenu(materialsManager))
    { }

    UIStyle& uil::UIManager::getStyle()
    {
        return style;
    }

    void uil::UIManager::togglePage(MenuPage page)
    {
        currentPage = currentPage == page ? MenuPage::Home : page;
    }

    void uil::UIManager::renderHome()
    {
        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        float sidebarWidth = expanded ? style.sidebarWidth : 0.0f;

        // Sidebar body
        if (expanded)
        {
            ImGui::SetNextWindowPos(ImVec2(displaySize.x - sidebarWidth, 0));
            ImGui::SetNextWindowSize(ImVec2(sidebarWidth, displaySize.y));

            ImGui::Begin(
                "Sidebar",
                nullptr,
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoSavedSettings
            );

            // Make image button and classes?
            if (ImGui::Selectable("Tools"))
            {
                togglePage(MenuPage::Tools);
                extraWidth = style.toolsWidth;
            }

            if (ImGui::Selectable("Materials"))
            {
                togglePage(MenuPage::Materials);
                extraWidth = style.materialsWidth;
            }

            ImGui::End();
        }

        // Pull tab
        ImGui::SetNextWindowPos(
            ImVec2(displaySize.x - sidebarWidth - style.pulltabWidth - (currentPage == MenuPage::Home || !expanded ? 0 : extraWidth), displaySize.y * 0.5f - style.pulltabHeight * 0.5f)
        );

        ImGui::SetNextWindowSize(ImVec2(style.pulltabWidth, style.pulltabHeight));

        ImGui::Begin(
            "SidebarTab",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoSavedSettings
        );

        ImGui::SetCursorPos(ImVec2(2.0f, 4.0f));

        if (ImGui::Button(expanded ? ">" : "<", ImVec2(style.pulltabWidth - 4.0f, style.pulltabHeight - 8.0f)))
        {
            expanded = !expanded;
        }

        ImGui::End();
    }

    void uil::UIManager::render()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        renderHome();

        if (expanded)
        {
            ImVec2 displaySize = ImGui::GetIO().DisplaySize;

            switch (currentPage)
            {
            case MenuPage::Materials:
                materialsMenu.render(style, ImVec2(displaySize.x - style.sidebarWidth - style.materialsWidth, 0));
                break;
            }
        }
    }
}