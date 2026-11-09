#include <uil/MaterialsManager.h>
#include <cstdint>
#include <string>
#include <functional>
#include <array>
#include <vector>
#include <imgui.h>

namespace uil
{
    Material::Material(uint32_t materialTypeIndex) : materialTypeIndex(materialTypeIndex), params{}
    {
    }

    Material Material::clone()
    {
        Material newMat(materialTypeIndex);

        for (size_t i = 0; i < params.size(); i++)
        {
            newMat.params[i] = params[i];
        }

        return newMat;
    }

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

    void MaterialsManager::addMaterialType(std::string name, std::vector<InputData> inputs, std::vector<Conversion> conversions)
    {
        materialTypes.push_back(MaterialType(materialTypes.size(), inputs, conversions));
        typeNames.push_back(name);
    }

    void MaterialsManager::createMaterial(uint32_t type)
    {
        MaterialType materialType = materialTypes[type];
        Material mat = materialType.initializeMaterial();
        MaterialInstance node(items.size(), mat);
        materialLayouts[items.size()] = (materialType.toMaterialLayout(mat));
        items.push_back(node);
        setDirty();
    }

    void MaterialsManager::cloneMaterial(uint32_t index)
    {
        Material clonedMat = items[index].mat.clone();
        MaterialInstance clonedInstance(items.size(), clonedMat);
        materialLayouts[items.size()] = materialTypes[clonedMat.materialTypeIndex].toMaterialLayout(clonedMat);
        items.push_back(clonedInstance);
        setDirty();
    }

    MaterialInstance::MaterialInstance(uint32_t id, Material mat) :
        mat(std::move(mat)),
        name("material " + std::to_string(id)),
        id(id)
    {
    }

    bool MaterialInstance::renderInstance(uint32_t& uiId, std::vector<MaterialType>& materialTypes, std::vector<std::string>& typeNames)
    {
        bool changed = false;

        ImGui::PushID(uiId);
        uiId++;
        if (ImGui::TreeNode((name + "###00").c_str()))
        {
            uint32_t selectedIndex = mat.materialTypeIndex;

            ImGui::Text("Material Type");
            ImGui::SameLine();
            if (ImGui::BeginCombo("##Material Type", typeNames[selectedIndex].c_str()))
            {
                for (size_t i = 0; i < typeNames.size(); i++)
                {
                    bool isSelected = (selectedIndex == i);

                    ImGui::PushID(uiId);
                    uiId++;
                    if (ImGui::Selectable(typeNames[i].c_str(), isSelected))
                    {
                        // If i is different than selected index, change material
                        if (i != selectedIndex)
                        {
                            mat = materialTypes[i].initializeMaterial();
                            changed = true;
                        }
                        selectedIndex = i;
                    }
                    ImGui::PopID();

                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::PushID(uiId);
            uiId++;

            strncpy(buf, name.c_str(), sizeof(buf) - 1);

            ImGui::Text("Name");
            ImGui::SameLine();
            ImGui::InputText("##Name", buf, sizeof(buf));
            name = buf;
            ImGui::PopID();

            // Add material type slider
            for (size_t i = 0; i < materialTypes[mat.materialTypeIndex].inputs.size(); i++)
            {
                changed |= materialTypes[mat.materialTypeIndex].inputs[i].renderInput(&mat.params[i], uiId);
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
        uint32_t flags,
        std::string description)
        :
        inputType(inputType),
        inputName(std::move(inputName)),
        flags(flags),
        param1(param1),
        param2(param2),
        defaultValue(defaultValue),
        description(std::move(description))
    {
    }

    bool InputData::renderInput(void* param, uint32_t& uiId)
    {
        bool changed = false;

        ImGui::Text((inputName + " = ").c_str());
        if (description != "")
        {
            ImGui::SetItemTooltip(description.c_str());
        }
        ImGui::SameLine();

        ImGui::PushID(uiId);

        switch (inputType)
        {
        case InputType::INPUT_FLOAT:
            changed |= ImGui::InputFloat(("##" + inputName).c_str(), (float*)param, param1, param2, "%.3f", flags);
            break;
        case InputType::SLIDER_FLOAT:
            changed |= ImGui::SliderFloat(("##" + inputName).c_str(), (float*)param, param1, param2, "%.3f", flags);
            break;
        case InputType::INPUT_INT:
            changed |= ImGui::InputInt(("##" + inputName).c_str(), (int*)param, param1, param2, flags);
            break;
        case InputType::SLIDER_INT:
            changed |= ImGui::SliderInt(("##" + inputName).c_str(), (int*)param, param1, param2, "%d", flags);
            break;
        }
        ImGui::PopID();

        uiId++;

        return changed;
    }
}