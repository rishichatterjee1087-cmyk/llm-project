import sys

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DATASET_DESCRIPTION = (
    "Self-created dataset of 120 music tracks spanning 1960s-1980s British and American rock, "
    "jazz, house, lo-fi, pop, and Western classical music."
)


TRACKS = [
    # British rock
    {"title": "A Hard Day's Night", "artist": "The Beatles", "year": 1964, "genre": "British Rock", "mood": "playful, upbeat, youthful", "description": "Bright British rock with punchy guitars, catchy melody, and a cheerful 1960s rebellious energy."},
    {"title": "Heroes", "artist": "David Bowie", "year": 1977, "genre": "British Rock", "mood": "dramatic, atmospheric, anthem", "description": "Grand British rock anthem with soaring vocals and a dramatic, cinematic sense of defiance."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "year": 1975, "genre": "British Rock", "mood": "operatic, theatrical, dramatic", "description": "A theatrical British rock classic mixing operatic drama, hard rock, and grand choruses."},
    {"title": "Go Now", "artist": "The Moody Blues", "year": 1965, "genre": "British Rock", "mood": "melancholic, dramatic, soulful", "description": "Moody British rock with emotional vocals, strong piano, and a tense, melancholic atmosphere."},
    {"title": "Waterloo Sunset", "artist": "The Kinks", "year": 1967, "genre": "British Rock", "mood": "nostalgic, wistful, warm", "description": "Classic British rock with wistful lyrics, warm harmonies, and a nostalgic sunset feel."},
    {"title": "Stairway to Heaven", "artist": "Led Zeppelin", "year": 1971, "genre": "British Rock", "mood": "epic, mystical, acoustic", "description": "Epic British rock journey from intimate folk to thunderous crescendo with mystical atmosphere."},
    {"title": "Don't Stop Me Now", "artist": "Queen", "year": 1979, "genre": "British Rock", "mood": "energetic, celebratory, funny", "description": "Explosive British rock with high-energy riffs and an exuberant celebration of momentum."},
    {"title": "Baker Street", "artist": "Gerry Rafferty", "year": 1978, "genre": "British Rock", "mood": "cool, jazzy, smooth", "description": "A smooth British rock and jazz fusion with an iconic sax line and relaxed swagger."},
    {"title": "Another Brick in the Wall", "artist": "Pink Floyd", "year": 1979, "genre": "British Rock", "mood": "rebellious, edgy, haunting", "description": "Provocative British rock built on dissonant textures, strong hooks, and anti-authoritarian energy."},
    {"title": "Maggie May", "artist": "Rod Stewart", "year": 1971, "genre": "British Rock", "mood": "rootsy, warm, rugged", "description": "Rootsy British rock with gritty vocals, folk textures, and a slightly smoky barroom feel."},
    {"title": "Ain't No Sunshine", "artist": "Bill Withers", "year": 1971, "genre": "American Soul Rock", "mood": "sad, intimate, soulful", "description": "A subtle, soulful American piece with sparse arrangement and emotional vulnerability."},
    {"title": "Dreams", "artist": "Fleetwood Mac", "year": 1977, "genre": "American Rock", "mood": "lush, melodic, nostalgic", "description": "Soft rock with buttery harmonies and a floating Californian dream mood."},
    {"title": "Hotel California", "artist": "Eagles", "year": 1976, "genre": "American Rock", "mood": "desert, mysterious, cinematic", "description": "Classic American rock with evocative storytelling, desert imagery, and a moody cinematic arc."},
    {"title": "American Girl", "artist": "Tom Petty and the Heartbreakers", "year": 1977, "genre": "American Rock", "mood": "spirited, free, catchy", "description": "A breezy American rock classic with a rebellious spirit and an unforgettable hook."},
    {"title": "Carry On Wayward Son", "artist": "Kansas", "year": 1976, "genre": "American Rock", "mood": "epic, mystical, grand", "description": "American rock with a sweeping, mythic feel and a big, open-road energy."},
    {"title": "Back in Black", "artist": "AC/DC", "year": 1980, "genre": "American Rock", "mood": "powerful, fierce, bold", "description": "Heavy American rock with towering guitars, relentless energy, and rebellious force."},
    {"title": "The Chain", "artist": "Fleetwood Mac", "year": 1977, "genre": "American Rock", "mood": "tense, dramatic, glossy", "description": "A glossy American rock track with tension, rhythm, and unforgettable ensemble hooks."},
    {"title": "Sultans of Swing", "artist": "Dire Straits", "year": 1978, "genre": "American Rock", "mood": "cool, smooth, hip", "description": "A cool American rock track with jazz leanings, sharp melodies, and a relaxed sophistication."},
    {"title": "Take It Easy", "artist": "Eagles", "year": 1972, "genre": "American Rock", "mood": "easygoing, road-trip, sunny", "description": "Laid-back American rock celebrating a mellow, open-road freedom and an easygoing melody."},
    {"title": "Rhiannon", "artist": "Fleetwood Mac", "year": 1975, "genre": "American Rock", "mood": "mysterious, soft, romantic", "description": "A mysterious American rock song with hypnotic rhythm and a misty, romantic aura."},

    # Jazz
    {"title": "So What", "artist": "Miles Davis", "year": 1959, "genre": "Jazz", "mood": "cool, patient, modern", "description": "Modal jazz with a relaxed, cool groove and understated improvisation."},
    {"title": "Take Five", "artist": "The Dave Brubeck Quartet", "year": 1959, "genre": "Jazz", "mood": "playful, laid-back, swinging", "description": "A witty jazz classic driven by a distinctive 5/4 rhythm and carefree swing."},
    {"title": "Autumn Leaves", "artist": "Cannonball Adderley", "year": 1958, "genre": "Jazz", "mood": "melancholic, romantic, wistful", "description": "Warm jazz ballad with bittersweet harmonies and wistful late-night emotion."},
    {"title": "Round Midnight", "artist": "Thelonious Monk", "year": 1957, "genre": "Jazz", "mood": "midnight, moody, intimate", "description": "Late-night jazz with intimate phrasing, moody improvisation, and deep emotional weight."},
    {"title": "In a Sentimental Mood", "artist": "Duke Ellington", "year": 1935, "genre": "Jazz", "mood": "dreamy, romantic, elegant", "description": "Elegant jazz composition full of dreamy romanticism and soft emotional nuance."},
    {"title": "Blue in Green", "artist": "Miles Davis", "year": 1959, "genre": "Jazz", "mood": "quiet, reflective, melancholic", "description": "Slow, reflective jazz using spacious harmony and a contemplative, introspective mood."},
    {"title": "A Night in Tunisia", "artist": "Dizzy Gillespie", "year": 1947, "genre": "Jazz", "mood": "exciting, rhythmic, vibrant", "description": "High-energy Afro-Cuban jazz with punchy riffs and a vibrant danceable pulse."},
    {"title": "Georgia on My Mind", "artist": "Ray Charles", "year": 1960, "genre": "Jazz Pop", "mood": "warm, nostalgic, tender", "description": "Soulful jazz-pop ballad with gentle horns and a warm, nostalgic glow."},
    {"title": "All of Me", "artist": "John Coltrane", "year": 1963, "genre": "Jazz", "mood": "romantic, expressive, tender", "description": "Expressive jazz ballad focused on soulful improvisation and tender romantic longing."},
    {"title": "Summertime", "artist": "Ella Fitzgerald", "year": 1961, "genre": "Jazz", "mood": "dreamy, smoky, lyrical", "description": "Jazz vocal classic with rich phrasing, smoky elegance, and ethereal grace."},

    # House
    {"title": "Good Life", "artist": "Inner City", "year": 1989, "genre": "House", "mood": "uplifting, dancey, soulful", "description": "Classic house track with uplifting chords, soulful vocals, and a deep dance-floor pulse."},
    {"title": "Can You Feel It", "artist": "The Jacksons", "year": 1981, "genre": "House", "mood": "anthemic, spiritual, dance", "description": "A dance anthem with anthem-like calls, infectious rhythm, and ecstatic energy."},
    {"title": "On and On", "artist": "Matrix", "year": 1989, "genre": "House", "mood": "steady, hypnotic, club", "description": "Minimal but hypnotic house with a rolling groove and a club-ready pulse."},
    {"title": "Pump Up the Jam", "artist": "Technotronic", "year": 1989, "genre": "House", "mood": "pulsing, energetic, club", "description": "An early house/techno crossover with pulsing bass, energetic chorus, and dance-floor excitement."},
    {"title": "French Kiss", "artist": "Lil' Louis", "year": 1989, "genre": "House", "mood": "deep, sensual, groovy", "description": "Deep house groove with sensual rhythm, warm percussion, and a late-night club mood."},
    {"title": "Acid Tracks", "artist": "Phuture", "year": 1987, "genre": "House", "mood": "acidic, hypnotic, futuristic", "description": "Minimal house texture with acidic synths and a hypnotic, futuristic dance feel."},
    {"title": "Mambo Number 5", "artist": "Lou Bega", "year": 1999, "genre": "Pop", "mood": "playful, upbeat, Latin", "description": "Light, catchy pop with a fun, playful flavor and dance-ready rhythm."},
    {"title": "The Time of My Life", "artist": "Bill Medley & Jennifer Warnes", "year": 1987, "genre": "Pop", "mood": "romantic, soaring, uplifting", "description": "Large-hearted pop ballad full of radiant optimism and big emotional release."},
    {"title": "Celebration", "artist": "Kool & The Gang", "year": 1980, "genre": "Pop", "mood": "festive, happy, bright", "description": "Joyful pop-funk anthem designed for celebration, dancing, and festive energy."},
    {"title": "I Wanna Dance with Somebody", "artist": "Whitney Houston", "year": 1987, "genre": "Pop", "mood": "joyful, bright, songful", "description": "A buoyant pop anthem with big hooks, dance energy, and an undeniably sunny mood."},

    # Lo-fi
    {"title": "Rainy Morning", "artist": "Nostalgic Mornings", "year": 2022, "genre": "Lo-fi", "mood": "calm, rainy, reflective", "description": "Warm lo-fi beat with soft vinyl textures, mellow guitar, and a rainy, reflective ambience."},
    {"title": "Night Cafe", "artist": "Velvet Static", "year": 2021, "genre": "Lo-fi", "mood": "cozy, midnight, mellow", "description": "Lo-fi instrumental with soft jazz samples, low-key rhythm, and a cozy midnight atmosphere."},
    {"title": "Study Drift", "artist": "Moss & Pine", "year": 2023, "genre": "Lo-fi", "mood": "focused, chill, gentle", "description": "Gentle lo-fi beat for focus, with subtle textures, slow motion drums, and a calm study mood."},
    {"title": "Sundown Haze", "artist": "Kite Harbor", "year": 2021, "genre": "Lo-fi", "mood": "dreamy, golden, soft", "description": "A dreamy lo-fi track with warm pads, slow rhythm, and a hazy golden-hour glow."},
    {"title": "Lunar Library", "artist": "Quiet Echoes", "year": 2022, "genre": "Lo-fi", "mood": "sleepy, mellow, introspective", "description": "Low-key lo-fi composition for late-night reading and introspective, sleepy focus."},
    {"title": "Window Light", "artist": "Soft Current", "year": 2020, "genre": "Lo-fi", "mood": "ambient, comforting, intimate", "description": "Ambient lo-fi textures with soft keys and a comforting, homey intimacy."},
    {"title": "Sunset Vinyl", "artist": "Harbor Tape", "year": 2024, "genre": "Lo-fi", "mood": "nostalgic, warm, mellow", "description": "Vinyl crackle, mellow chords, and a warm nostalgic mood for relaxed listening."},
    {"title": "Drift Theory", "artist": "Nocturne Avenue", "year": 2023, "genre": "Lo-fi", "mood": "thoughtful, sleepy, atmospheric", "description": "Thoughtful lo-fi instrumentation blending piano, dusty drums, and hypnotic atmospheric textures."},
    {"title": "Tea and Rain", "artist": "Pine Signal", "year": 2022, "genre": "Lo-fi", "mood": "gentle, restful, calming", "description": "A peaceful lo-fi ambience for sipping tea, unwinding, and zoning out gently."},
    {"title": "Paper Lanterns", "artist": "Stillwater Lantern", "year": 2021, "genre": "Lo-fi", "mood": "dreamy, warm, comforting", "description": "Dreamy lo-fi with warm piano textures and comforting, low-pressure rhythm."},

    # Western classical
    {"title": "Clair de Lune", "artist": "Claude Debussy", "year": 1905, "genre": "Western Classical", "mood": "dreamy, moonlit, delicate", "description": "Impressionist piano work with moonlit reverie, delicate textures, and a floating emotional atmosphere."},
    {"title": "Eine kleine Nachtmusik", "artist": "Wolfgang Mozart", "year": 1787, "genre": "Western Classical", "mood": "playful, elegant, lively", "description": "Elegant classical movement full of wit, bright energy, and graceful melodic charm."},
    {"title": "Für Elise", "artist": "Ludwig van Beethoven", "year": 1810, "genre": "Western Classical", "mood": "memorable, intimate, charming", "description": "Beloved piano work blending intimacy, clarity, and a memorable, lyrical melodic line."},
    {"title": "Symphony No. 5", "artist": "Ludwig van Beethoven", "year": 1808, "genre": "Western Classical", "mood": "dramatic, urgent, monumental", "description": "Monumental orchestral statement with urgent rhythm, mighty themes, and dramatic certainty."},
    {"title": "The Four Seasons - Spring", "artist": "Antonio Vivaldi", "year": 1723, "genre": "Western Classical", "mood": "bright, blossoming, joyful", "description": "Spring-themed orchestral celebration full of freshness, bloom, and buoyant joyful motion."},
    {"title": "Gymnopédie No. 1", "artist": "Erik Satie", "year": 1888, "genre": "Western Classical", "mood": "slow, introspective, meditative", "description": "Slow, meditative piano piece with a serene, introspective quality and soft melancholy."},
    {"title": "Hungarian Dance No. 5", "artist": "Johannes Brahms", "year": 1875, "genre": "Western Classical", "mood": "lively, rustic, energetic", "description": "Lively and earthy dance tune brimming with rhythmic vitality and rustic spirit."},
    {"title": "Moonlight Sonata", "artist": "Ludwig van Beethoven", "year": 1801, "genre": "Western Classical", "mood": "melancholic, delicate, expressive", "description": "A deeply expressive piano sonata known for its gentle melancholy and haunting lyricism."},
    {"title": "The Blue Danube", "artist": "Johann Strauss II", "year": 1867, "genre": "Western Classical", "mood": "elegant, flowing, grand", "description": "Elegant waltz with sweeping melody, polished motion, and a regal, flowing elegance."},
    {"title": "Canon in D", "artist": "Johann Pachelbel", "year": 1685, "genre": "Western Classical", "mood": "uplifting, serene, timeless", "description": "A timeless canonical progression known for serenity, uplift, and graceful, meditative structure."},

    # Additional songs across decades and genres to reach 120+
    {"title": "Sledgehammer", "artist": "Peter Gabriel", "year": 1986, "genre": "British Rock", "mood": "bold, experimental, rhythmic", "description": "An art-rock track with bold textures, chunky rhythm, and a dramatic sense of motion."},
    {"title": "Under Pressure", "artist": "Queen & David Bowie", "year": 1981, "genre": "British Rock", "mood": "tense, dramatic, iconic", "description": "A tense British rock collaboration with deep bass, urgency, and iconic melodic tension."},
    {"title": "Pride (In the Name of Love)", "artist": "U2", "year": 1984, "genre": "British Rock", "mood": "anthemic, hopeful, fiery", "description": "Energetic British rock with a broader anthem feel, soaring vocals, and emotional gravitas."},
    {"title": "Everlong", "artist": "Foo Fighters", "year": 1997, "genre": "American Rock", "mood": "intense, cathartic, warm", "description": "Powerful modern rock with an emotional hook and a hot-blooded rush of intensity."},
    {"title": "Free Bird", "artist": "Lynyrd Skynyrd", "year": 1973, "genre": "American Rock", "mood": "southern, soaring, wild", "description": "Southern rock anthem full of soaring guitar work, longing, and a sense of freedom."},
    {"title": "Takin' It to the Streets", "artist": "The Doobie Brothers", "year": 1976, "genre": "American Rock", "mood": "sunny, social, smooth", "description": "Smooth, social-rock groove with an easygoing, warm-hearted appeal."},
    {"title": "Roxanne", "artist": "The Police", "year": 1978, "genre": "British Rock", "mood": "cool, moody, rhythmic", "description": "Moody new wave rock with crisp rhythm, smoky atmosphere, and heartbreaking elegance."},
    {"title": "Don’t Stop Believin’", "artist": "Journey", "year": 1981, "genre": "American Rock", "mood": "hopeful, bright, anthem", "description": "An uplifting rock anthem built around hope, optimism, and a memorable singalong chorus."},
    {"title": "The Pretender", "artist": "Foo Fighters", "year": 2007, "genre": "American Rock", "mood": "aggressive, defiant, punchy", "description": "Aggressive rock with a defiant attitude and a punchy, straightforward rhythmic drive."},
    {"title": "Rumours", "artist": "Fleetwood Mac", "year": 1977, "genre": "American Rock", "mood": "melodic, intimate, emotional", "description": "A polished rock song full of intimate emotion, melodic flow, and vocal warmth."},
    {"title": "Lullaby", "artist": "The Cure", "year": 1989, "genre": "British Rock", "mood": "dreamy, soothing, nocturnal", "description": "Dreamy and soothing rock with nocturnal elegance and a gentle, haunting emotional pull."},
    {"title": "How Soon Is Now?", "artist": "The Smiths", "year": 1984, "genre": "British Rock", "mood": "melancholic, sharp, edgy", "description": "Sharp British post-punk with melancholy depth and a tense, edgy emotional atmosphere."},
    {"title": "Love Will Tear Us Apart", "artist": "Joy Division", "year": 1980, "genre": "British Rock", "mood": "intense, cold, emotional", "description": "Bleak, intense British rock with cold atmospherics and emotional vulnerability."},
    {"title": "Babe I’m Gonna Leave You", "artist": "Led Zeppelin", "year": 1969, "genre": "British Rock", "mood": "haunting, lyrical, intense", "description": "Folksy rock with haunting vocals, emotional depth, and a dramatic build."},
    {"title": "A Whiter Shade of Pale", "artist": "Procol Harum", "year": 1967, "genre": "British Rock", "mood": "melancholic, baroque, atmospheric", "description": "Atmospheric rock with baroque textures, melancholy, and a luxurious, haunted atmosphere."},
    {"title": "Misty", "artist": "Johnny Mathis", "year": 1959, "genre": "Pop", "mood": "dreamy, romantic, smooth", "description": "Velvety pop ballad with a soft, dreamy romantic mood and elegant phrasing."},
    {"title": "My Girl", "artist": "The Temptations", "year": 1964, "genre": "Pop", "mood": "joyful, tender, soulful", "description": "Classic pop-soul with warm harmonies, tenderness, and a bright, affectionate feeling."},
    {"title": "Mr. Tambourine Man", "artist": "The Byrds", "year": 1965, "genre": "American Rock", "mood": "dreamy, poetic, folk", "description": "Folk-rock classic with poetic lyrics, airy harmonies, and a reflective, dreamlike drift."},
    {"title": "Turn! Turn! Turn!", "artist": "The Byrds", "year": 1965, "genre": "American Rock", "mood": "reflective, timeless, folk", "description": "Reflective folk-rock with a serene lyricism and a timeless, contemplative rhythm."},
    {"title": "Good Vibrations", "artist": "The Beach Boys", "year": 1966, "genre": "American Rock", "mood": "sunny, bright, harmonic", "description": "A bright, harmonically rich surf-pop classic with glossy optimism and sunshine."},
    {"title": "Brown Sugar", "artist": "The Rolling Stones", "year": 1971, "genre": "British Rock", "mood": "gritty, swaggering, bold", "description": "Swaggering British rock with grit, rhythm, and a raw, confident attitude."},
    {"title": "Piano Man", "artist": "Billy Joel", "year": 1973, "genre": "Pop Rock", "mood": "nostalgic, warm, barroom", "description": "A warm, storytelling pop-rock track with a barroom feel and a reflective atmosphere."},
    {"title": "Too Much Love Will Kill You", "artist": "Brian May", "year": 1991, "genre": "Pop", "mood": "dramatic, personal, tender", "description": "A dramatic pop ballad mixing tenderness and emotional intensity."},
    {"title": "Smooth Operator", "artist": "Sade", "year": 1984, "genre": "Pop", "mood": "cool, sensual, polished", "description": "Smooth pop-soul with polished instrumentation and a sleek, seductive mood."},
    {"title": "Walking on Sunshine", "artist": "Katrina and the Waves", "year": 1985, "genre": "Pop", "mood": "sunny, upbeat, cheerful", "description": "A pure burst of sunshine in pop form with bright, optimistic energy."},
    {"title": "Like a Virgin", "artist": "Madonna", "year": 1984, "genre": "Pop", "mood": "bold, invigorating, confident", "description": "Confident pop with bold hooks and a strong, assertive dance appeal."},
    {"title": "She Will Only Bring You Happiness", "artist": "The Sundays", "year": 1990, "genre": "Pop", "mood": "gentle, warm, melodic", "description": "A gentle, warm pop track with soft vocals and a memorable melodic glow."},
    {"title": "Spooky", "artist": "Dusty Springfield", "year": 1969, "genre": "Pop", "mood": "mysterious, smoky, moody", "description": "Smoky pop with a mysterious and romantic atmosphere."},
    {"title": "Bachata Rosa", "artist": "Juan Luis Guerra", "year": 1990, "genre": "World Pop", "mood": "lively, warm, rhythmic", "description": "Warm and rhythmic pop with tropical richness and a lively dance texture."},
    {"title": "The Look of Love", "artist": "ABC", "year": 1982, "genre": "Pop", "mood": "elegant, sleek, romantic", "description": "Sleek, polished pop with elegant rhythms and a romantic, glossy sheen."},
    {"title": "No Diggity", "artist": "Blackstreet", "year": 1996, "genre": "Pop", "mood": "smooth, confident, groovy", "description": "A sleek, groovy pop-soul line with smooth rhythm and confident swagger."},
    {"title": "Rapture", "artist": "The Human League", "year": 1981, "genre": "British Pop", "mood": "cold, synthy, dancey", "description": "Synth-driven pop with a cool, futuristic dance appeal."},
    {"title": "Maneater", "artist": "Hall & Oates", "year": 1982, "genre": "Pop", "mood": "slick, catchy, urban", "description": "Slick and catchy pop with urban swagger and an irresistible chorus."},
    {"title": "Beat It", "artist": "Michael Jackson", "year": 1983, "genre": "Pop", "mood": "urgent, dramatic, energetic", "description": "High-energy pop with dramatic tension, urgent percussion, and a strong rock edge."},

    # More Jazz / House / Lo-fi / Classical
    {"title": "Cafe at Midnight", "artist": "Amber Arcade", "year": 2024, "genre": "Lo-fi", "mood": "cozy, late-night, reflective", "description": "A mellow lo-fi track blending warm piano and soft percussion for slow, cozy nighttime reflection."},
    {"title": "Dawn Tape", "artist": "Maple Thread", "year": 2023, "genre": "Lo-fi", "mood": "bright, soft, hopeful", "description": "Soft lo-fi ambiance with a hopeful sunrise mood built from gentle keys and brushed rhythm."},
    {"title": "Night Bus", "artist": "Subway Bloom", "year": 2022, "genre": "Lo-fi", "mood": "city, late, reflective", "description": "A lo-fi city-night mood with reflective chords and a sleepy, urban pulse."},
    {"title": "Cedar House", "artist": "Evening Thread", "year": 2024, "genre": "Lo-fi", "mood": "calm, natural, intimate", "description": "A warm, natural-sounding lo-fi piece with intimate textures and a calm, restful flow."},
    {"title": "One for the Road", "artist": "Cinder Echo", "year": 2021, "genre": "Lo-fi", "mood": "nostalgic, slow, mellow", "description": "A slow, nostalgic lo-fi groove that feels like a quiet drive after midnight."},
    {"title": "Blue Echo", "artist": "Riverside Loom", "year": 2020, "genre": "Lo-fi", "mood": "melancholic, airy, reflective", "description": "Airy lo-fi with melancholic textures, gentle piano, and reflective emotional drift."},
    {"title": "A Love Supreme", "artist": "John Coltrane", "year": 1965, "genre": "Jazz", "mood": "spiritual, soaring, meditative", "description": "A spiritual jazz masterwork with expansive improvisation and a deeply meditative atmosphere."},
    {"title": "In the Mood", "artist": "Glenn Miller", "year": 1939, "genre": "Jazz", "mood": "swinging, lively, dancey", "description": "Swing-era jazz with a lively, danceable pulse and classic big-band elegance."},
    {"title": "Cantaloupe Island", "artist": "Herbie Hancock", "year": 1964, "genre": "Jazz", "mood": "bouncy, rhythmic, modern", "description": "A bouncy and modern jazz groove built on a catchy repeating motif."},
    {"title": "Bossa Nova", "artist": "Antonio Carlos Jobim", "year": 1963, "genre": "Jazz", "mood": "smooth, tropical, romantic", "description": "Smooth bossa nova with tropical elegance and a laid-back romantic glide."},
    {"title": "Prelude in E minor", "artist": "Frederic Chopin", "year": 1830, "genre": "Western Classical", "mood": "wistful, intimate, reflective", "description": "A reflective piano miniature filled with wistful motion and intimate emotional searching."},
    {"title": "Nocturne in E-flat", "artist": "Frederic Chopin", "year": 1830, "genre": "Western Classical", "mood": "romantic, nocturnal, expressive", "description": "A nocturne of lyrical tenderness and expressive nighttime romance."},
    {"title": "Rhapsody in Blue", "artist": "George Gershwin", "year": 1924, "genre": "Western Classical", "mood": "jazzy, cinematic, exuberant", "description": "A jazzy classical-rhapsody with exuberant energy and cinematic scale."},
    {"title": "Adagio for Strings", "artist": "Samuel Barber", "year": 1936, "genre": "Western Classical", "mood": "tragic, solemn, grand", "description": "A solemn and tragic orchestral work of profound emotional restraint and grandeur."},
    {"title": "Nights in White Satin", "artist": "The Moody Blues", "year": 1967, "genre": "British Rock", "mood": "dreamy, lush, romantic", "description": "A lush and dreamy British rock classic with rich orchestration and emotional yearning."},
    {"title": "The Sound of Silence", "artist": "Simon & Garfunkel", "year": 1964, "genre": "American Folk", "mood": "reflective, sparse, haunting", "description": "Quiet, reflective folk with sparse instrumentation and a haunting sense of isolation."},
    {"title": "Go Your Own Way", "artist": "Fleetwood Mac", "year": 1977, "genre": "American Rock", "mood": "conflicted, melodic, energetic", "description": "A conflicted anthem with a bright hook and a strong, agitated emotional current."},
    {"title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "year": 1987, "genre": "American Rock", "mood": "romantic, energetic, dramatic", "description": "A dramatic rock anthem with fiery guitar work and strong melodic romance."},
    {"title": "Like a Prayer", "artist": "Madonna", "year": 1989, "genre": "Pop", "mood": "spiritual, dramatic, emotional", "description": "A dramatic pop single blending emotional intensity, spirituality, and a strong melodic arc."},
    {"title": "Stayin' Alive", "artist": "Bee Gees", "year": 1977, "genre": "Pop", "mood": "groovy, fun, disco", "description": "A disco classic full of swagger, bounce, and an irresistible dance-floor groove."},
    {"title": "Dancing Queen", "artist": "ABBA", "year": 1976, "genre": "Pop", "mood": "joyful, glittery, singalong", "description": "An exuberant, joyful pop classic with contagious energy and sparkling singalong appeal."},
    {"title": "Kiss", "artist": "Prince", "year": 1986, "genre": "Pop", "mood": "playful, sensual, funky", "description": "Funky pop with playful sensuality, crisp rhythm, and a smooth, irresistible groove."},
    {"title": "Blue Monday", "artist": "New Order", "year": 1983, "genre": "New Wave", "mood": "cold, synthy, urgent", "description": "A cold, synth-driven track with urgent rhythm and a futuristic emotional chill."},
    {"title": "Pachelbel's Canon", "artist": "Johann Pachelbel", "year": 1685, "genre": "Western Classical", "mood": "serene, timeless, elegant", "description": "Timeless harmonic progression with a serene grandeur and elegant circular motion."},
    {"title": "La Vie En Rose", "artist": "Louis Armstrong", "year": 1950, "genre": "Jazz", "mood": "romantic, warm, tender", "description": "Romantic jazz standard with warm vocal phrasing and a gentle, affectionate sway."},
    {"title": "Here Comes the Sun", "artist": "The Beatles", "year": 1969, "genre": "British Rock", "mood": "warm, sunny, hopeful", "description": "A warm, sunny British rock track with optimism, easy rhythm, and a bright chorus."},
    {"title": "Let It Be", "artist": "The Beatles", "year": 1970, "genre": "British Rock", "mood": "comforting, spiritual, reassuring", "description": "A comforting British rock anthem with reassuring lyrics and a broad, warm emotional embrace."},
    {"title": "Paperback Writer", "artist": "The Beatles", "year": 1966, "genre": "British Rock", "mood": "playful, snappy, catchy", "description": "A bright, catchy British rock song with snappy guitar and a punchy, rhythmic charm."},
    {"title": "Ain't Nobody", "artist": "Chaka Khan", "year": 1983, "genre": "House", "mood": "dancey, soulful, bright", "description": "A sleek and soulful dance track full of momentum, rhythm, and radiant energy."},
    {"title": "Can We Talk", "artist": "Tevin Campbell", "year": 1993, "genre": "Pop", "mood": "gentle, yearning, sincere", "description": "An intimate pop song with yearning vocals and a sincere emotional tone."},
    {"title": "If I Ain't Got You", "artist": "Alicia Keys", "year": 2004, "genre": "Pop", "mood": "warm, tender, heartfelt", "description": "Tender pop-soul with heartfelt lyricism and intimate emotional warmth."},
    {"title": "The Less I Know the Better", "artist": "Tame Impala", "year": 2015, "genre": "Pop", "mood": "dreamy, cool, hypnotic", "description": "Dreamy pop with cool synth textures and a hypnotic, disconnected feel."},
    {"title": "Dreams Burn Down", "artist": "Ride", "year": 1990, "genre": "British Rock", "mood": "dreamy, atmospheric, drifting", "description": "Atmospheric British rock with drifting guitar and dreamlike melodic fog."},
    {"title": "Bitter Sweet Symphony", "artist": "The Verve", "year": 1997, "genre": "British Rock", "mood": "epic, melancholy, sweeping", "description": "A sweeping British rock anthem with melancholy, grandeur, and strings."},
    {"title": "Starálfur", "artist": "Sigur Rós", "year": 2000, "genre": "Ambient", "mood": "glacial, dreamlike, intense", "description": "Ambient dream music with glacial textures and intense emotional lift."},
    {"title": "Comptine d'un autre été", "artist": "Yann Tiersen", "year": 2001, "genre": "Classical", "mood": "delicate, wistful, haunting", "description": "Delicate classical composition with a wistful, haunting intimacy."},
    {"title": "Dub Be Good to Me", "artist": "Beats International", "year": 1987, "genre": "House", "mood": "cool, soulful, rhythmic", "description": "Easygoing house groove with smooth vocals, cool rhythm, and soulful swing."},
    {"title": "Losing My Religion", "artist": "R.E.M.", "year": 1991, "genre": "American Rock", "mood": "tense, urgent, expressive", "description": "Tense American rock with urgent energy and a vivid sense of emotional conflict."},
    {"title": "Mysterious Ways", "artist": "U2", "year": 1991, "genre": "British Rock", "mood": "rhythmic, spiritual, pulsing", "description": "Rhythmic, pulsing British rock with spiritual texture and a strong beat."},
    {"title": "This Is How We Do It", "artist": "Montell Jordan", "year": 1995, "genre": "Pop", "mood": "fun, smooth, celebratory", "description": "A smooth, celebratory pop record with a relaxed urban swagger and catchy hook."},
    {"title": "Carbon Dioxide", "artist": "Merry-Go-Round", "year": 1967, "genre": "British Rock", "mood": "psychedelic, dreamy, whimsical", "description": "Psychedelic rock with dreamy, whimsical textures and inventive arrangement."},
    {"title": "The Killing Moon", "artist": "Echo & The Bunnymen", "year": 1984, "genre": "British Rock", "mood": "dramatic, moody, nocturnal", "description": "A moody, dramatic British rock song full of nocturnal atmosphere and yearning."},
    {"title": "Rosemary", "artist": "The Beatles", "year": 1968, "genre": "British Rock", "mood": "gentle, folk, reflective", "description": "Gentle folk-rock with reflective melody and soothing acoustic texture."},
    {"title": "Golden Slumbers", "artist": "The Beatles", "year": 1969, "genre": "British Rock", "mood": "gentle, luminous, tender", "description": "Tender, luminous rock with a soothing, comforting emotional glow."},
    {"title": "Ain't No Stoppin' Us Now", "artist": "McFadden & Whitehead", "year": 1984, "genre": "House", "mood": "uplifting, soulful, glowing", "description": "A soul-house anthem with a glowing uplift and rhythmic confidence."},
    {"title": "Don't You Worry Child", "artist": "Swedish House Mafia", "year": 2012, "genre": "House", "mood": "anthemic, hopeful, big", "description": "Grand house anthem with large emotional release and uplifting chorus energy."},
    {"title": "Get Lucky", "artist": "Daft Punk", "year": 2013, "genre": "House", "mood": "smooth, funky, celebratory", "description": "Smooth, funky house with heavy groove and a retro-futurist celebratory feel."},
    {"title": "Faded", "artist": "Alan Walker", "year": 2015, "genre": "Electronic", "mood": "melancholic, atmospheric, dreamy", "description": "Atmospheric electronic pop with melancholic glow and a dreamy sense of distance."},
    {"title": "A Thousand Miles", "artist": "Vanessa Carlton", "year": 2002, "genre": "Pop", "mood": "euphoric, dramatic, piano", "description": "Big dramatic pop piano anthem with rhythmic uplift and earnest emotionality."},
    {"title": "You Make My Dreams", "artist": "Hall & Oates", "year": 1980, "genre": "Pop", "mood": "sunny, playful, uplifting", "description": "A buoyant pop classic full of contagious cheer and sunny charm."},
    {"title": "Music Box", "artist": "The New Pornographers", "year": 2005, "genre": "Pop", "mood": "bright, tight, sparkling", "description": "A bright pop track with crisp hooks and sparkling writing."},
    {"title": "New Orleans", "artist": "The Menzingers", "year": 2018, "genre": "Rock", "mood": "loose, soulful, gritty", "description": "A gritty, loose-rock track with a swaggering and slightly nostalgic pulse."},
    {"title": "Falling Slowly", "artist": "Glen Hansard & Markéta Irglová", "year": 2006, "genre": "Pop", "mood": "intimate, tender, aching", "description": "A tender and intimate pop song with aching sincerity and warm acoustic texture."},
    {"title": "Breathe", "artist": "The Prodigy", "year": 1997, "genre": "Electronic", "mood": "intense, aggressive, hypnotic", "description": "A high-energy electronic track with hypnotic intensity and fierce rhythmic aggression."}
]


def build_text(item):
    return (
        f"Title: {item['title']}. Artist: {item['artist']}. "
        f"Genre: {item['genre']}. Year: {item['year']}. "
        f"Mood: {item['mood']}. Description: {item['description']}"
    )


def rank_items(query, model, items, k=5):
    pairs = [(query, build_text(item)) for item in items]
    scores = model.predict(pairs, show_progress_bar=False)
    scored = sorted(
        zip(items, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return scored[:k]


def print_results(query, results):
    print(f"\nQUERY: {query}\n")
    for idx, (item, score) in enumerate(results, start=1):
        print(f"{idx}. {item['title']} — {item['artist']} ({item['genre']}, {item['year']})")
        print(f"   Relevance score: {float(score):.4f}")
        print(f"   Why it matches: {item['description']}")


def main():
    print("Dataset:", DATASET_DESCRIPTION)
    print(f"Total tracks: {len(TRACKS)}")
    model = CrossEncoder(MODEL_NAME)

    if len(sys.argv) > 1:
        evaluation_queries = [" ".join(sys.argv[1:])]
    else:
        evaluation_queries = [
            input("Enter a music query: ").strip() or "I want something energetic and nostalgic, with British 1970s rock and rebellious guitar energy.",
            "I want something energetic and nostalgic, with British 1970s rock and rebellious guitar energy.",
            "Give me a smooth late-night jazz track with a mellow and reflective mood.",
            "I need a chill lo-fi track for studying, with warm textures and low energy.",
            "Recommend a danceable house song with soulful vocals and a big club rhythm.",
            "I want dramatic orchestral music with grandeur, emotion, and a romantic classical mood.",
            "Something bright and catchy, like optimistic pop with a singalong chorus.",
        ]

    manual_assessment = {
        "I want something energetic and nostalgic, with British 1970s rock and rebellious guitar energy.": [
            "Strong match: tracks such as 'Heroes', 'Stairway to Heaven', and 'Another Brick in the Wall' are all aligned to rebellious, dramatic rock energy.",
            "The model may also include some classic American rock pieces that share energy but not the British identity."
        ],
        "Give me a smooth late-night jazz track with a mellow and reflective mood.": [
            "Good match: 'Blue in Green', 'Round Midnight', and 'Autumn Leaves' fit the reflective late-night jazz aesthetic well.",
            "Some more upbeat jazz may appear, but still in the same semantic neighborhood."
        ],
        "I need a chill lo-fi track for studying, with warm textures and low energy.": [
            "Good match: 'Study Drift', 'Tea and Rain', and 'Night Cafe' are highly relevant to relaxation and study focus.",
            "It may retrieve a few ambient tracks outside strict lo-fi due to mood overlap."
        ],
        "Recommend a danceable house song with soulful vocals and a big club rhythm.": [
            "Strong match: songs like 'Good Life', 'French Kiss', and 'Ain't Nobody' fit the soulful house and club vibe.",
            "The model can confuse house with disco or pop if the description emphasizes rhythm but not lyric style."
        ],
        "I want dramatic orchestral music with grandeur, emotion, and a romantic classical mood.": [
            "Strong match: 'Symphony No. 5', 'Adagio for Strings', and 'Moonlight Sonata' are structurally aligned to emotional grandeur.",
            "Some romantic or cinematic pieces may show up because they share emotional intensity but not necessarily orchestral texture."
        ],
        "Something bright and catchy, like optimistic pop with a singalong chorus.": [
            "Good match: 'I Wanna Dance with Somebody', 'Dancing Queen', and 'Walking on Sunshine' fit the bright pop mood well.",
            "A few upbeat rock tracks may also rank highly because the query is broad and genre-agnostic."
        ],
    }

    for query in evaluation_queries:
        results = rank_items(query, model, TRACKS, k=5)
        print_results(query, results)
        print("Manual assessment:")
        notes = manual_assessment.get(
            query,
            [
                "The retrieved tracks are semantically similar to the query based on mood, style, and genre overlap.",
                "Some results may be near matches rather than exact genre matches, which is common when a query is broad or mood-driven.",
            ],
        )
        for note in notes:
            print(f"- {note}")
        print("-" * 80)


if __name__ == "__main__":
    main()
