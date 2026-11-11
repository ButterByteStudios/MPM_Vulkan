class ToolsManager
{
public:
	bool isDirty();
	void setDirty();
	void setClean();

	void getToolStruct();

private:

	bool dirty = false; // Can be changed to a layout version int64 if there are multiple consumers (like for rendering)
};