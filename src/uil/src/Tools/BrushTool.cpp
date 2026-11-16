#include <uil/Tool.h>

#define _USE_MATH_DEFINES
#include <math.h>

namespace uil
{    
    class BrushTool : public Tool
    {
    public:
        void update(float deltaTime) override;
        void handleInput() override
        {
            seed++;
        }

        const char* getName() const override
        {
            return "Brush";
        }

        ToolGPUData getGPUData() const override
        {
            BrushToolData gpuData{};

            gpuData.pos = cellPos;
            gpuData.previousPos = previousCellPos;
            gpuData.brushSize = brushSize;
            gpuData.brushShape = brushShape;


            float distance = glm::length(cellPos - previousCellPos);
            float brushArea = shapeArea();

            float diameter = 2.0f * brushSize;
            float rectangleArea = diameter * distance;
            float areaFactor = (rectangleArea) / brushArea;

            gpuData.particleCount = static_cast<uint32_t>(static_cast<float>(particleDensity) * brushArea * areaFactor);
            gpuData.seed = seed;
            //gpuData.mass = 

            return gpuData;
        }

        float shapeArea() const
        {
            switch (brushShape)
            {
            case 0:
                return M_PI * brushSize * brushSize;
            case 1:
                return 4.0f * brushSize * brushSize;
            }
        }

    private:

        glm::vec2 cellPos;
        glm::vec2 previousCellPos;
        float brushSize;
        uint32_t materialId;
        uint32_t particleDensity;
        uint32_t brushShape;
        uint32_t seed = 0;

        static constexpr const char* brushShapes[] = { "Disk", "Box" };
    };
}