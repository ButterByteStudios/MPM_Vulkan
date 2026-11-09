#pragma once

#include <uil/MaterialsMenu.h>
#include <uil/UIStyle.h>

namespace uil
{
	enum class MenuPage
	{
		Home,
		Tools,
		Materials
	};

	class ToolsMenu
	{

	};

	class UIManager
	{
	public:
		UIManager(MaterialsManager& materialsManager);

		void render();

		void renderHome();

		void toggleExpanded();

		void togglePage(MenuPage page);

		UIStyle& getStyle();

	private:
		UIStyle style;
		MenuPage currentPage = MenuPage::Home;
		bool expanded = false;
		float extraWidth = 0;

		MaterialsMenu materialsMenu;
	};
}