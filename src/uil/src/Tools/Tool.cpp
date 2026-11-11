namespace uil
{
	class Tool
	{
	public:
		virtual ~Tool() = default;

		virtual void update(float deltaTime) = 0;
		virtual void handleInput() = 0;

		//virtual ToolGPUData getGPUData() const = 0

	};
}