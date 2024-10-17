from random import randrange
import aiohttp
import discord
from discord.ext import commands
import funny_flame
import cv2

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

origin_cost = (
    (0, 0), (0, 0), (1, 30), (2, 65), (3, 105), (5, 150), (7, 200), (9, 255), (12, 315),
    (15, 380), (25, 580), (28, 660), (31, 750), (35, 850), (39, 960), (43, 1080), (47, 1210),
    (51, 1350), (55, 1500), (60, 1660), (75, 2010), (80, 2180), (85, 2360), (90, 2550),
    (95, 2750), (100, 2960), (106, 3180), (112, 3410), (118, 3650), (125, 3900), (145, 4400)
)

mastery_cost = (
    (0, 0), (3, 50), (4, 65), (5, 83), (6, 103), (7, 126), (8, 151), (9, 179), (11, 209),
    (13, 242), (18, 342), (20, 382), (22, 427), (24, 477), (26, 532), (28, 592), (30, 657),
    (32, 727), (34, 802), (37, 882), (45, 1057), (48, 1142), (51, 1232), (54, 1327), (57, 1427),
    (60, 1532), (63, 1642), (66, 1757), (69, 1877), (73, 2002), (83, 2252)
)

enhancement_cost = (
    (0, 0), (4, 75), (5, 98), (6, 125), (7, 155), (9, 189), (11, 227), (13, 269),
    (16, 314), (19, 363), (27, 513), (30, 573), (33, 641), (36, 716), (39, 799),
    (42, 889), (45, 987), (48, 1092), (51, 1205), (55, 1325), (67, 1588), (71, 1716),
    (75, 1851), (79, 1994), (83, 2144), (87, 2302), (92, 2467), (97, 2640), (102, 2820),
    (108, 3008), (123, 3383)
)

common_cost = (
    (0, 0), (7, 125), (9, 163), (11, 207), (13, 257), (16, 314), (19, 377), (22, 446),
    (27, 521), (32, 603), (46, 903), (51, 1013), (56, 1137), (62, 1275), (68, 1427),
    (74, 1592), (80, 1771), (86, 1964), (92, 2171), (99, 2391), (116, 2916), (123, 3150),
    (130, 3398), (137, 3660), (144, 3935), (151, 4224), (160, 4527), (169, 4844), (178, 5174),
    (188, 5518), (208, 6268)
)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} commands')
        # await bot.guild_commands()
    except Exception as e:
        print(e)


@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')


@bot.command()
async def marco(ctx):
    await ctx.send('Polo!')


@bot.command()
async def rps(ctx):
    bot_hand = ['Rock', 'Paper', 'Scissor']
    num_hand = randrange(3)
    print(num_hand)
    bot_select = bot_hand[num_hand]
    print(bot_select)
    user_message = ctx.message.content
    user_message = user_message.replace('!rps', '')
    user_message = user_message.lower()
    print(user_message)

    if user_message == ' rock' or user_message == ' paper' or user_message == ' scissor':
        await ctx.send(f'EasyErda chooses {bot_select}')

    if user_message == ' rock':
        if bot_select == 'Rock':
            await ctx.send('Tie!')
        elif bot_select == 'Paper':
            await ctx.send('You Lost! HEHE')
        elif bot_select == 'Scissor':
            await ctx.send('Diu, You Won.')

    if user_message == ' paper':
        if bot_select == 'Rock':
            await ctx.send('Diu, You Won.')
        elif bot_select == 'Paper':
            await ctx.send('Tie!')
        elif bot_select == 'Scissor':
            await ctx.send('You Lost! HEHE')

    if user_message == ' scissor':
        if bot_select == 'Rock':
            await ctx.send('Diu, You Won.')
        elif bot_select == 'Paper':
            await ctx.send('Diu, You Won.')
        elif bot_select == 'Scissor':
            await ctx.send('Tie!')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        # ignore discord bot messages
        return

    if message.content:
        if message.content[0] == '!':
            # Process commands
            message.content = message.content.lower()
            await bot.process_commands(message)
            return

    if len(message.attachments) > 1:
        await message.channel.send('Please only send one image at a time.')
        return

    if message.attachments:
        for attachment in message.attachments:
            content_type = attachment.content_type

        # await message.channel.send(f'Type: This is a {content_type}')
        if content_type and content_type.startswith('image/'):
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()

                        # Save the image locally
                        with open('discord_user_image.png', 'wb') as f:
                            f.write(image_data)

                        # Read the image using OpenCV
                        image = cv2.imread('discord_user_image.png')

                        #funny_flame.flame_process('discord_user_image.png')

                        if image is not None:
                            # Process the image
                             await message.reply(funny_flame.flame_process('discord_user_image.png'))
                        else:
                            await message.channel.send('Failed to read the image using OpenCV.')

    else:
        pass
        # await message.channel.send(f'Type: This is a user message')


@bot.tree.command(name='hello')
async def hello(interaction: discord.Interaction):
    embed = discord.Embed(title="Title", description="This is an embedded message.", color=0x00ff00)
    embed.add_field(name="Field1", value="Some value", inline=False)
    await interaction.response.send_message(embed=embed)
    #await interaction.response.send_message(f'Hi')


@bot.tree.command(name='wap')
async def wap(interaction: discord.Interaction, coretype: str, currentlevel: int, targetlevel: int):
    if currentlevel <= -1:
        await interaction.response.send_message(f'Error: Current level cannot be lower than 0.')
        return
    if targetlevel >= 31:
        await interaction.response.send_message(f'Error: Target level cannot be above 30.')
        return
    if currentlevel > targetlevel:
        await interaction.response.send_message(f'Error: Current level cannot be higher than target level.')
        return
    if currentlevel == targetlevel:
        await interaction.response.send_message(f'Error: Current level cannot be the same as target level.')
        return

    try:
        if coretype == 'origin':
            result = tuple(a - b for a, b in zip(origin_cost[targetlevel], origin_cost[currentlevel]))
            await interaction.response.send_message(
                f'Origin Core Lv.{currentlevel} to Lv.{targetlevel} will cost you:\n{result[0]} Sol Erdas and {result[1]} Fragments\nBetter start WAPing!')
        elif coretype == 'enhance':
            result = tuple(a - b for a, b in zip(enhancement_cost[targetlevel], enhancement_cost[currentlevel]))
            await interaction.response.send_message(
                f'Enhancement Core Lv.{currentlevel} to Lv.{targetlevel} will cost you:\n{result[0]} Sol Erdas and {result[1]} Fragments\nBetter start WAPing!')
        elif coretype == 'mastery':
            result = tuple(a - b for a, b in zip(mastery_cost[targetlevel], mastery_cost[currentlevel]))
            await interaction.response.send_message(
                f'Mastery Core Lv.{currentlevel} to Lv.{targetlevel} will cost you:\n{result[0]} Sol Erdas and {result[1]} Fragments\nBetter start WAPing!')
        elif coretype == 'common':
            result = tuple(a - b for a, b in zip(common_cost[targetlevel], common_cost[currentlevel]))
            await interaction.response.send_message(
                f'Common Core Lv.{currentlevel} to Lv.{targetlevel} will cost you:\n{result[0]} Sol Erdas and {result[1]} Fragments\nBetter start WAPing!')
        else:
            await interaction.response.send_message(
                f'coreType must be one of the following: origin, enhance, mastery, or common')
    except Exception as e:
        await interaction.response.send_message(f'Error processing input: {str(e)}')


@bot.tree.command(name='progress')
async def progress(interaction: discord.Interaction, origin: str, enhance: str, mastery: str, common: str):
    try:
        section = {
            'origin': origin,
            'enhance': enhance,
            'mastery': mastery,
            'common': common
        }

        args = {name: eval(section[name]) for name in section}
        keys = list(args.keys())
        origin_solfrag = (0, 0)
        enhance_solfrag = (0, 0)
        mastery_solfrag = (0, 0)
        common_solfrag = (0, 0)
        for i in keys:
            print(args[i])
            if isinstance(args[i], int):
                args[i] = tuple([args[i]])
            for j in args[i]:
                if i == 'origin':
                    origin_solfrag = tuple(a + b for a, b in zip(origin_solfrag, origin_cost[j]))
                elif i == 'enhance':
                    enhance_solfrag = tuple(a + b for a, b in zip(enhance_solfrag, enhancement_cost[j]))
                elif i == 'mastery':
                    mastery_solfrag = tuple(a + b for a, b in zip(mastery_solfrag, mastery_cost[j]))
                elif i == 'common':
                    common_solfrag = tuple(a + b for a, b in zip(common_solfrag, common_cost[j]))
        # Create response string with named categories
        response = '\n'.join([f'{name.capitalize()}: {args[name]} '
                              f'Sol Used: {origin_solfrag[0] if name == "origin" else (enhance_solfrag[0] if name == "enhance" else (mastery_solfrag[0] if name == "mastery" else common_solfrag[0]))}'
                              f', Fragments Used: {origin_solfrag[1] if name == "origin" else (enhance_solfrag[1] if name == "enhance" else (mastery_solfrag[1] if name == "mastery" else common_solfrag[1]))}'
                              for name in args])

        if origin_solfrag[0] + enhance_solfrag[0] + mastery_solfrag[0] + common_solfrag[0] == 0:
            await interaction.response.send_message('https://tenor.com/view/where-you-at-gif-21177622')
            return

        # Send the processed results as a response
        await interaction.response.send_message(f'Processed results:\n\n{response}'
                                                f'\n\nTotal Sol Erda: {origin_solfrag[0] + enhance_solfrag[0] + mastery_solfrag[0] + common_solfrag[0]}'
                                                f'\nTotal Sol Fragments: {origin_solfrag[1] + enhance_solfrag[1] + mastery_solfrag[1] + common_solfrag[1]}')

    except Exception as e:
        await interaction.response.send_message(f'Error processing input: {str(e)}')


def run_bot():
    TOKEN = ''
    bot.run(TOKEN)
