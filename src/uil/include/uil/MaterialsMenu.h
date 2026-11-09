#pragma once

#include <uil/MaterialsManager.h>
#include <uil/UIStyle.h>

namespace uil
{
	class MaterialsMenu
	{
	public:
		MaterialsMenu(MaterialsManager& manager);

		void render(UIStyle& style, ImVec2 position);

	private:
		MaterialsManager& manager;
	};
}