#include <imgui.h>
#include <materials.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <utility>

#include <cstdint>
#include <string>
#include <functional>
#include <array>
#include <vector>

// Add bounds checks for safety

Material::Material(uint32_t materialTypeIndex): materialTypeIndex(materialTypeIndex), params{}
{ }

MaterialsManager::MaterialsManager(uint32_t maxMaterialCount) : maxMaterialCount(maxMaterialCount), materialLayouts{}
{
    materialLayouts.resize(maxMaterialCount);
}

MaterialsManager::MaterialsManager()
{
}

bool MaterialsManager::isDirty()
{
    return dirty;
}

void MaterialsManager::setDirty()
{
    dirty = true;
}

void MaterialsManager::setClean()
{
    dirty = false;
}

void MaterialsManager::updateMaterialLayout(uint32_t index)
{
    // Update materials
    MaterialLayout mat = materialTypes[items[index].mat.materialTypeIndex].toMaterialLayout(items[index].mat);
    materialLayouts[index] = mat;
    setDirty();
}

void MaterialsManager::addMaterialType(std::vector<InputData> inputs, std::vector<Conversion> conversions)
{
    materialTypes.push_back(MaterialType(materialTypes.size(), inputs, conversions));
}

void MaterialsManager::addMaterial(uint32_t type)
{
    MaterialType materialType = materialTypes[type];
    Material mat = materialType.initializeMaterial();
    MaterialInstance node(items.size(), mat);
    materialLayouts[items.size()] = (materialType.toMaterialLayout(mat));
    items.push_back(node);
    setDirty();
}

void MaterialsManager::render()
{
    uint32_t uiId = 0;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();

    ImGui::Begin("Materials demo");

    static int selectedIndex = -1;
    if (ImGui::BeginListBox("##00"))
    {
        for (int n = 0; n < static_cast<uint32_t>(items.size()); n++)
        {
            ImGui::PushID(uiId);
            uiId++;
            if (items[n].renderInstance(uiId, materialTypes[items[n].mat.materialTypeIndex]))
            {
                updateMaterialLayout(n);
            }

            ImGui::PopID();
        }

        ImGui::EndListBox();
    }

    if (static_cast<uint32_t>(items.size()) < maxMaterialCount && ImGui::Button("Add"))
    {
        addMaterial(0);
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

MaterialInstance::MaterialInstance(uint32_t id, Material mat) : 
    mat(std::move(mat)),
    name("material " + std::to_string(id)),
    id(id)
{
}

bool MaterialInstance::renderInstance(uint32_t &uiId, MaterialType& materialType)
{
    bool changed = false;

    ImGui::PushID(uiId);
    uiId++;
    if (ImGui::TreeNode((name + "###00").c_str()))
    {
        ImGui::PushID(uiId);
        uiId++;

        strncpy(buf, name.c_str(), sizeof(buf) - 1);
        ImGui::InputText("Name", buf, sizeof(buf));
        name = buf;
        ImGui::PopID();

        // Add material type slider
        for (size_t i = 0; i < materialType.inputs.size(); i++)
        {
            changed |= materialType.inputs[i].renderInput(&mat.params[i], uiId);
        }

        ImGui::TreePop();
    }
    ImGui::PopID();

    return changed;
}

MaterialType::MaterialType(uint32_t index, std::vector<InputData> inputs, std::vector<Conversion> conversions) : 
    index(index),
    inputs(std::move(inputs)),
    conversions(std::move(conversions))
{
}

MaterialLayout MaterialType::toMaterialLayout(const Material& mat)
{
    MaterialLayout layout{};
    layout.materialType = index;

    for (size_t i = 0; i < conversions.size(); i++)
    {
        layout.params[i] = conversions[i](mat.params);
    }

    return layout;
}

Material MaterialType::initializeMaterial()
{
    Material mat(index);
    for (size_t i = 0; i < inputs.size(); i++)
    {
        mat.params[i] = inputs[i].defaultValue;
    }

    return mat;
}

InputData::InputData(
    std::string inputName,
    InputType inputType,
    float param1,
    float param2,
    float defaultValue,
    uint32_t flags)
    :
    inputType(inputType),
    inputName(std::move(inputName)),
    flags(flags),
    param1(param1),
    param2(param2),
    defaultValue(defaultValue)
{
}

bool InputData::renderInput(void* param, uint32_t &uiId)
{
    bool changed = false;

    ImGui::Text((inputName + " = ").c_str());
    ImGui::SameLine();

    ImGui::PushID(uiId);

    switch (inputType)
    {
        case InputType::INPUT_FLOAT:
            changed |= ImGui::InputFloat(("##" + inputName).c_str(), (float*)param, param1, param2);
            break;
        case InputType::SLIDER_FLOAT:
            changed |= ImGui::SliderFloat(("##" + inputName).c_str(), (float*)param, param1, param2);
            break;
        case InputType::INPUT_INT:
            changed |= ImGui::InputInt(("##" + inputName).c_str(), (int*)param, param1, param2);
            break;
        case InputType::SLIDER_INT:
            changed |= ImGui::SliderInt(("##" + inputName).c_str(), (int*)param, param1, param2);
            break;
    }
    ImGui::PopID();

    uiId++;

    return changed;
}