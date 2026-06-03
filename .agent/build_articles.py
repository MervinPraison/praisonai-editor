#!/usr/bin/env python3
"""Build biblerevelation sermon articles — scannable format, PPT verses + transcript."""

import json
import re
from pathlib import Path

# Soft sky-blue — distinct from PPT gold (#FFD700)
_VERSE_MARK = (
    '<mark style="background-color:#DBEAFE;color:#1E3A5F;'
    'padding:0 2px;border-radius:2px;">'
)
_AGENT = Path(__file__).parent
_ACTIVE_HL: dict[str, list[str]] = {}


def norm_ref(ref: str) -> str:
    r = ref.replace("\u2013", "-").replace("\u2014", "-")
    r = re.sub(r"\s*-\s*(opening verse|Part \d+/\d+).*$", "", r, flags=re.I)
    r = re.sub(r"\s*\([^)]*\)", "", r)
    r = re.sub(r"[^\w\s:\-]", "", r)
    return re.sub(r"\s+", " ", r).strip().lower()


def load_highlights(name: str) -> dict[str, list[str]]:
    path = _AGENT / name
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for item in data:
        key = norm_ref(item.get("ref", ""))
        if not key:
            continue
        phrases: list[str] = []
        for v in item.get("verses", []):
            phrases.extend(v.get("emph", []))
        out.setdefault(key, []).extend(phrases)
    return out


FOLD_HL = load_highlights("100_fold_highlights.json")
DELAY_HL = load_highlights("why_delay_highlights.json")


def apply_highlights(text: str, phrases: list[str]) -> str:
    if not phrases:
        return text
    result = text
    for phrase in sorted(set(phrases), key=len, reverse=True):
        if not phrase.strip():
            continue
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f"{_VERSE_MARK}{m.group(0)}</mark>", result
        )
    return result


def sep():
    return """<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->"""


def h2(text):
    return f"""<!-- wp:heading -->
<h2 class="wp-block-heading">{text}</h2>
<!-- /wp:heading -->"""


def h3(text):
    return f"""<!-- wp:heading {{"level":3}} -->
<h3 class="wp-block-heading">{text}</h3>
<!-- /wp:heading -->"""


def para(text):
    return f"""<!-- wp:paragraph -->
<p>{text}</p>
<!-- /wp:paragraph -->"""


def hebrew_name_changes():
    """Large Hebrew names — added Hey (ה) highlighted in amber."""
    m = (
        '<mark style="background-color:#FDE68A;color:#78350F;'
        'padding:2px 6px;border-radius:4px;">'
    )
    e = "</mark>"
    return f"""<!-- wp:paragraph {{"align":"center"}} -->
<p class="has-text-align-center" style="font-size:2.25em;line-height:1.8;font-family:'Times New Roman',serif;">
<span style="direction:rtl;display:inline-block;">אַבְרָם → אַבְרָ{m}הָ{e}ם</span><br/>
<span style="direction:rtl;display:inline-block;">שָׂרַי → שָׂר{m}ה{e}ָ</span>
</p>
<!-- /wp:paragraph -->"""


def quote(text):
    return f"""<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>{text}</p></blockquote>
<!-- /wp:quote -->"""


def verse(ref, text):
    """Verse reference as bold label + blockquote — PPT phrase highlights applied."""
    emph = _ACTIVE_HL.get(norm_ref(ref), [])
    body = apply_highlights(text, emph)
    return f"""{para(f"<strong>{ref}</strong>")}
{quote(body)}"""


def ul(items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f"""<!-- wp:list -->
<ul class="wp-block-list">{lis}</ul>
<!-- /wp:list -->"""


def big_emoji(text):
    """Large centred emoji for table cells."""
    return (
        f'<span style="font-size:3em;line-height:1.2;display:block;'
        f'text-align:center;">{text}</span>'
    )


def table(headers, rows, align=None):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    if align == "full":
        opener = '<!-- wp:table {"align":"full"} -->'
        fig_class = "wp-block-table alignfull"
    elif align == "wide":
        opener = '<!-- wp:table {"align":"wide"} -->'
        fig_class = "wp-block-table alignwide"
    else:
        opener = "<!-- wp:table -->"
        fig_class = "wp-block-table"
    return f"""{opener}
<figure class="{fig_class}"><table style="width:100%"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></figure>
<!-- /wp:table -->"""


def image(media_id, full_url, src_url, alt=""):
    """Reuse existing WordPress media — no re-upload."""
    alt_attr = f' alt="{alt}"' if alt else ' alt=""'
    return f"""<!-- wp:image {{"id":{media_id},"sizeSlug":"large","linkDestination":"media","align":"wide"}} -->
<figure class="wp-block-image alignwide size-large"><a href="{full_url}"><img src="{src_url}"{alt_attr} class="wp-image-{media_id}"/></a></figure>
<!-- /wp:image -->"""


# Reused from https://biblerevelation.org/2026/05/the-gospel-in-the-stars/ (post 240179)
GOSPEL_STARS_BIBLE = image(
    240183,
    "https://biblerevelation.org/wordpress/wp-content/uploads/2026/05/The-Gospel-in-the-Stars-Bible.png",
    "https://biblerevelation.org/wordpress/wp-content/uploads/2026/05/The-Gospel-in-the-Stars-Bible-1024x1024.png",
    "The Gospel in the Stars — Virgo to Leo (Bible)",
)
GOSPEL_STARS_WHEEL = image(
    240185,
    "https://biblerevelation.org/wordpress/wp-content/uploads/2026/05/The-Gospel-in-the-Stars.png",
    "https://biblerevelation.org/wordpress/wp-content/uploads/2026/05/The-Gospel-in-the-Stars-1024x1024.png",
    "The Gospel in the Stars — zodiac gospel chart",
)


def section(h2t, *blocks):
    return "\n\n".join([sep(), h2(h2t), *blocks])


def section_h3(h3t, *blocks):
    return "\n\n".join([sep(), h3(h3t), *blocks])


def build_100_fold():
    global _ACTIVE_HL
    _ACTIVE_HL = FOLD_HL
    blocks = [
        h3("🌾 100 Fold Blessings — Four Levels of Faith"),
        para(
            "Scripture speaks of <mark><strong>hundredfold</strong></mark> fruit — "
            "<em>not at random, but for you</em>. When Isaac sowed, he reaped a hundredfold. "
            "A little time with God can multiply a hundred times when you believe it."
        ),
        ul([
            "🎯 <strong>Goal:</strong> good ground → 30, 60, or <mark>100 fold</mark>",
            "🛤️ <strong>Path:</strong> 4 levels of faith (see table below)",
            "📖 <strong>Key:</strong> hear the word, understand it, bear fruit",
        ]),
        table(
            ["Level", "Name", "What happens", "Key verse"],
            [
                ["1️⃣", "Starting faith", "Word snatched — believe <em>truth</em>, not facts", "Matthew 13:19"],
                ["2️⃣", "Faith tested", "Tribulation — stay rooted; keep hearing", "James 1:3"],
                ["3️⃣", "Keeping faith", "Thorns / cares — hold confession of hope", "Hebrews 10:23"],
                ["4️⃣", "Victory", "Overcome + bear fruit; touch others", "1 John 5:4"],
            ],
        ),
        para("<strong>Matthew 13 — Parable of the Sower at a glance:</strong>"),
        table(
            ["#", "", "In short", "Where seed falls", "What happens (parable)", "What it means", "Faith level", "Outcome"],
            [
                [
                    "1",
                    big_emoji("🏜️"),
                    "Sand only — seed devoured",
                    "<strong>Wayside</strong>",
                    "Fell by the wayside; birds devoured them (13:4)",
                    "Hears the word but <em>does not understand</em> — wicked one snatches it (13:19)",
                    "Starting faith",
                    "Word stolen — no root",
                ],
                [
                    "2",
                    big_emoji("🌱"),
                    "Tiny sprout — no root",
                    "<strong>Stony places</strong>",
                    "No depth of earth; sprang up, then scorched — no root, withered (13:5–6)",
                    "Receives with joy but <em>no root</em>; tribulation → stumbles (13:20–21)",
                    "Faith tested",
                    "Withers under pressure",
                ],
                [
                    "3",
                    big_emoji("🪴"),
                    "Rooted — no fruit",
                    "<strong>Among thorns</strong>",
                    "Thorns sprang up and choked them (13:7)",
                    "Cares of this world + deceitfulness of riches <em>choke</em> the word (13:22)",
                    "Keeping faith",
                    "Unfruitful",
                ],
                [
                    "4",
                    big_emoji("🌳🍎"),
                    "Rooted — bears fruit",
                    "<strong>Good ground</strong>",
                    "Yielded a crop — hundredfold, sixty, thirty (13:8)",
                    "Hears, <em>understands</em>, bears fruit — hundredfold, sixty, thirty (13:23)",
                    "Victory",
                    "30 · 60 · <mark>100 fold</mark>",
                ],
            ],
            align="full",
        ),
        section(
            "📖 The Goal — Good Ground",
            para("Hear → understand → produce <strong>thirty, sixty, or hundredfold</strong>."),
            verse(
                "Matthew 13:23 (NKJV)",
                "But he who received seed on the good ground is he who hears the word and understands it, who indeed bears fruit and produces: some a hundredfold, some sixty, some thirty.",
            ),
        ),
        section(
            "🔥 Pentecost — Greater Works",
            ul([
                "Jesus said <strong>most assuredly</strong> — not maybe",
                "You will do what He did — and <mark><strong>more</strong></mark>",
                "Tongues → same power Jesus had (Acts pattern)",
                "Two results: do His works + <u>greater</u> works",
            ]),
            verse(
                "John 14:12 (NKJV)",
                "Most assuredly, I say to you, he who believes in Me, the works that I do he will do also; and greater works than these he will do, because I go to My Father.",
            ),
        ),
        section(
            "✝️ Three Messages — Don't Mix Them",
            para(
                "The <strong>gospel = good news</strong>. Some verses speak mercy; others wrath. "
                "<em>Which covenant? To whom was it spoken?</em> Context is key."
            ),
            table(
                ["Voice", "Theme", "Example", "For whom?"],
                [
                    ["✝️ <strong>Jesus</strong>", "Mercy &amp; grace", "Psalm 103", "Believers — cross took wrath"],
                    ["📜 <strong>Moses</strong>", "Law &amp; deeds", "Romans 2:6, 13", "Doers of law"],
                    ["🔥 <strong>Elijah</strong>", "Wrath", "Romans 2:8", "Self-seeking, unbelieving"],
                ],
            ),
            para("<strong>Message of Jesus</strong>"),
            verse(
                "Psalm 103:10 (NKJV)",
                "He has not dealt with us according to our sins, Nor punished us according to our iniquities.",
            ),
            verse(
                "Psalm 103:8-9 (NKJV)",
                "The Lord is merciful and gracious, Slow to anger, and abounding in mercy. He will not always strive with us, Nor will He keep His anger forever.",
            ),
            para("<strong>Message of Moses</strong>"),
            verse(
                "Romans 2:13 (NKJV)",
                "(for not the hearers of the law are just in the sight of God, but the doers of the law will be justified;",
            ),
            verse(
                "Romans 2:6 (NKJV)",
                'who "will render to each one according to his deeds":',
            ),
            para("<strong>Message of Elijah</strong>"),
            verse(
                "Romans 2:8 (NKJV)",
                "but to those who are self-seeking and do not obey the truth, but obey unrighteousness—indignation and wrath,",
            ),
        ),
        section(
            "1️⃣ Level 1 — Starting Faith",
            table(
                ["Facts (what you feel)", "Truth (what Bible says)"],
                [
                    ["“I am sick”", "“By His stripes — <em>zero</em> pain is mine”"],
                    ["Sickness is “normal”", "Word snatched — devil steals seed"],
                ],
            ),
            ul([
                "Faith comes by <strong>hearing</strong> — again and again",
                "Don't let online “facts” steal the seed (Matthew 13:19)",
            ]),
            verse(
                "Romans 10:17 (NKJV)",
                "So then faith comes by hearing, and hearing by the word of God.",
            ),
            verse(
                "Matthew 13:4 (NKJV)",
                "And as he sowed, some seed fell by the wayside; and the birds came and devoured them.",
            ),
            verse(
                "Matthew 13:19 (NKJV)",
                "When anyone hears the word of the kingdom, and does not understand it, then the wicked one comes and snatches away what was sown in his heart. This is he who received seed by the wayside.",
            ),
        ),
        section(
            "2️⃣ Level 2 — Faith Tested",
            para("You believe healing — then symptoms appear. <mark>Don't switch to facts.</mark> Keep hearing the same verse until rooted."),
            verse(
                "James 1:3 (NKJV)",
                "knowing that the testing of your faith produces patience.",
            ),
            verse(
                "Matthew 13:5-6 (NKJV)",
                "Some fell on stony places, where they did not have much earth; and they immediately sprang up because they had no depth of earth. But when the sun was up they were scorched, and because they had no root they withered away.",
            ),
            verse(
                "Matthew 13:20-21 (NKJV)",
                "But he who received the seed on stony places, this is he who hears the word and immediately receives it with joy; yet he has no root in himself, but endures only for a while. For when tribulation or persecution arises because of the word, immediately he stumbles.",
            ),
        ),
        section(
            "3️⃣ Level 3 — Keeping the Faith",
            ul([
                "Cares of this world + riches choke the word",
                "Even when sick → speak <strong>truth</strong>, not facts",
                "Hold fast confession of hope <em>without wavering</em> (Peter, Paul model)",
            ]),
            verse(
                "2 Timothy 4:7 (NKJV)",
                "I have fought the good fight, I have finished the race, I have kept the faith.",
            ),
            verse(
                "Hebrews 10:23 (NKJV)",
                "Let us hold fast the confession of our hope without wavering, for He who promised is faithful.",
            ),
            verse(
                "Matthew 13:7 (NKJV)",
                "And some fell among thorns, and the thorns sprang up and choked them.",
            ),
            verse(
                "Matthew 13:22 (NKJV)",
                "Now he who received seed among the thorns is he who hears the word, and the cares of this world and the deceitfulness of riches choke the word, and he becomes unfruitful.",
            ),
        ),
        section(
            "4️⃣ Level 4 — Victory &amp; Fruitful Faith",
            ul([
                "You are <strong>born of God</strong> → overcome the world",
                "Level up: sickness over you → you conquer → others healed through you",
                "Fruit: <mark>30 · 60 · 100 fold</mark>",
            ]),
            verse(
                "1 John 5:4 (NKJV)",
                "For whatever is born of God overcomes the world. And this is the victory that has overcome the world—our faith.",
            ),
            verse(
                "Matthew 13:8 (NKJV)",
                "But others fell on good ground and yielded a crop: some a hundredfold, some sixty, some thirty.",
            ),
            verse(
                "Matthew 13:23 (NKJV)",
                "But he who received seed on the good ground is he who hears the word and understands it, who indeed bears fruit and produces: some a hundredfold, some sixty, some thirty.",
            ),
        ),
        sep(),
        h2("✅ Final Thought"),
        ul([
            "🌱 Starting → ⚡ Tested → 🙏 Keeping → 🏆 Victory",
            "Feeling sick but believing health? Say: <em>“My hundredfold is coming.”</em>",
            "Hear + speak the word again and again — you <u>will</u> see the hundredfold. Amen.",
        ]),
    ]
    return "\n\n".join(blocks)


def build_delay():
    global _ACTIVE_HL
    _ACTIVE_HL = DELAY_HL
    blocks = [
        h3("⏱️ How to Prevent Delay — Abraham's Instant Blessing"),
        verse(
            "Romans 4:3 (NKJV) — opening verse",
            "For what does the Scripture say? \"Abraham believed God, and it was accounted to him for righteousness.\"",
        ),
        para(
            "Pentecost Sunday — Romans 4 is about <strong>Abraham, faith, and righteousness</strong>. "
            "Two steps to instant blessing: (1) understand faith &amp; righteousness · (2) understand what changed at <strong>99</strong>."
        ),
        table(
            ["Pentecost", "When", "What happened"],
            [
                ["1st (Sinai)", "50 days after Egypt", "❌ 3000 <strong>died</strong>"],
                ["2nd (Upper room)", "50 days after ascension", "✅ 3000 <strong>saved</strong> (Peter preaching)"],
            ],
        ),
        para("Letter kills · Spirit gives life — Moses vs Jesus (Romans / 2 Corinthians 3)."),
        verse(
            "Matthew 7:11 (KJV)",
            "If ye then, being evil, know how to give good gifts unto your children: how much more shall your Father which is in heaven give good things to them that ask him?",
        ),
        section(
            "🌟 Abraham's Calling",
            ul([
                "Age <strong>75</strong> — first promise &amp; stars",
                "Age <strong>99</strong> — <em>El Shaddai</em> appears; delay ends",
                "Fully convinced God performs what He promised",
            ]),
            verse(
                "Genesis 12:2 (KJV)",
                "And I will make of thee a great nation, and I will bless thee, and make thy name great; and thou shalt be a blessing.",
            ),
            verse(
                "Genesis 17:1 (KJV)",
                "And when Abram was ninety years old and nine, the Lord appeared to Abram, and said unto him, I am the Almighty God; walk before me, and be thou perfect.",
            ),
            verse(
                "Romans 4:21 (NKJV)",
                "and being fully convinced that what He had promised He was also able to perform.",
            ),
            verse(
                "Romans 4:17 (NKJV)",
                '(as it is written, "I have made you a father of many nations") in the presence of Him whom he believed—God, who gives life to the dead and calls those things which do not exist as though they did;',
            ),
        ),
        section(
            "🍎 Two Fruits — Law vs Life",
            ul([
                "Eden: fruit of <em>good &amp; evil</em> vs fruit of <strong>eternal life</strong>",
                "Law = knowledge of good &amp; evil (old covenant — Romans)",
                "God's word = food · when you hear it, you eat spiritual fruit",
            ]),
            verse(
                "John 14:6 (KJV)",
                "Jesus saith unto him, I am the way, the truth, and the life: no man cometh unto the Father, but by me.",
            ),
            verse(
                "2 Corinthians 3:6 (NKJV)",
                "who also made us sufficient as ministers of the new covenant, not of the letter but of the Spirit; for the letter kills, but the Spirit gives life.",
            ),
            verse(
                "Romans 5:19 (NKJV)",
                "For as by one man's disobedience many were made sinners, so also by one Man's obedience many will be made righteous.",
            ),
        ),
        section(
            "📋 Part 1 — Age 75 to 85",
            para(
                "God promised a generation at <strong>75</strong>. Isaac came at <strong>99</strong> — "
                "25 years later. First understand <em>faith &amp; righteousness</em> (Part 1), "
                "then what changed at 99 (Part 2)."
            ),
        ),
        section_h3(
            "Characteristics of Abraham at age between 75 to 85",
            para(
                "The <em>first change</em> at 75 — stars shown, righteousness credited. "
                "Yet delay continued because Abraham still mixed <strong>his strength</strong> with God's."
            ),
            table(
                ["Characteristic", "Detail"],
                [
                    ["⭐ Stars &amp; gospel", "Believed the <strong>Seed (Christ)</strong> — Genesis 15"],
                    ["✝️ Righteousness", "Counted righteous <em>before</em> Isaac came"],
                    ["🔀 Mixed power", "Sufficient God — but <u>not all-sufficient</u> yet"],
                    ["🟢 70% self + 🔴 25% God", "Own effort + faith → <strong>delay</strong>"],
                    ["😨 Fear steps", "Sarai as <em>sister</em> · brought <strong>Lot</strong>"],
                ],
            ),
        ),
        section(
            "1️⃣ Knowing the Gospel",
            ul([
                "Gospel preached to Abraham <strong>before</strong> the law (Galatians 3:8)",
                "Word must be mixed with <em>faith</em> — or it doesn't profit (Hebrews 4:2)",
                "Israel: 40 years delay · Joshua: <u>instant</u> possession",
                "The <strong>Seed</strong> is Christ — one, not many (Galatians 3:16)",
            ]),
            para("<strong>⭐ Stars shown twice — meditate on the difference:</strong>"),
            table(
                ["When", "Reference", "Focus", "Faith &amp; righteousness?"],
                [
                    ["1st (age 75)", "Genesis 15:5–6", "The <strong>Seed (Christ)</strong> — gospel", "✅ Yes — counted righteous"],
                    ["2nd", "Genesis 22:17", "Generations / Isaac to come", "❌ Not mentioned — child promise"],
                ],
            ),
            ul([
                "Hebrew <strong>saphar (סָפַר)</strong> — “count the stars” also means <em>declare, tell, proclaim</em> the gospel",
                "Look at the stars → proclaim Christ's story (not astrology)",
                "True faith = believe Jesus died for <u>every iniquity</u> → that is your righteousness",
            ]),
            para("<strong>🌌 Gospel in the stars</strong> (Virgo → Leo — Christ's journey):"),
            ul([
                "♍ Virgin birth → ♎ price paid → ♏ enemy trampled → ♐ conquering rider",
                "♑ atoning sacrifice → ♒ living water → ♓ fishers of men → ♈ Lamb of God",
                "♉ Messiah's strength → ♊ God &amp; man → ♋ shepherd's care → ♌ <strong>Lion of Judah</strong>",
            ]),
            GOSPEL_STARS_BIBLE,
            verse(
                "Psalm 19:1 (KJV)",
                "The heavens declare the glory of God; and the firmament sheweth his handywork.",
            ),
            verse(
                "Luke 10:19 (KJV)",
                "Behold, I give unto you power to tread on serpents and scorpions, and over all the power of the enemy: and nothing shall by any means hurt you.",
            ),
            verse(
                "Revelation 5:5 (KJV)",
                "And one of the elders saith unto me, Weep not: behold, the Lion of the tribe of Juda, the Root of David, hath prevailed to open the book, and to loose the seven seals thereof.",
            ),
            verse(
                "Hebrews 4:2 (KJV)",
                "For unto us was the gospel preached, as well as unto them: but the word preached did not profit them, not being mixed with faith in them that heard it.",
            ),
            verse(
                "Hebrews 3:18-19 (KJV)",
                "And to whom sware he that they should not enter into his rest, but to them that believed not? So we see that they could not enter in because of unbelief.",
            ),
            verse(
                "Hebrews 4:1 (KJV)",
                "Let us therefore fear, lest, a promise being left us of entering into his rest, any of you should seem to come short of it.",
            ),
            verse(
                "Galatians 3:8 (KJV)",
                "And the scripture, foreseeing that God would justify the heathen through faith, preached before the gospel unto Abraham, saying, In thee shall all nations be blessed.",
            ),
            verse(
                "John 8:56 (KJV)",
                "Your father Abraham rejoiced to see my day: and he saw it, and was glad.",
            ),
            verse(
                "Genesis 15:5-6 (NKJV)",
                'Then He brought him outside and said, "Look now toward heaven, and count the stars if you are able to number them." And He said to him, "So shall your descendants be." And he believed in the LORD, and He accounted it to him for righteousness.',
            ),
            verse(
                "Genesis 22:17 (NKJV)",
                "I will bless you, and I will multiply your descendants as the stars of the heaven and as the sand which is on the seashore; and your descendants shall possess the gate of their enemies.",
            ),
            verse(
                "Galatians 3:16 (NKJV)",
                'Now to Abraham and his Seed were the promises made. He does not say, "And to seeds," as of many, but as of one, "And to your Seed," who is Christ.',
            ),
            verse(
                "Genesis 22:18 (NKJV)",
                "In your seed all the nations of the earth shall be blessed, because you have obeyed My voice.",
            ),
            verse(
                "Genesis 15:5-6 (KJV)",
                "And he brought him forth abroad, and said, Look now toward heaven, and tell the stars, if thou be able to number them: and he said unto him, So shall thy seed be. And he believed in the LORD; and he counted it to him for righteousness.",
            ),
            verse(
                "Romans 4:3 (KJV)",
                "For what saith the scripture? Abraham believed God, and it was counted unto him for righteousness.",
            ),
            verse(
                "Romans 4:3 (NKJV)",
                "For what does the Scripture say? \"Abraham believed God, and it was accounted to him for righteousness.\"",
            ),
        ),
        section(
            "2️⃣ Righteousness Apart from the Law",
            para("Promise = through <strong>faith</strong>, not law. Works not → believe → righteousness counted."),
            verse(
                "Romans 4:5 (KJV)",
                "But to him that worketh not, but believeth on him that justifieth the ungodly, his faith is counted for righteousness.",
            ),
            verse(
                "Romans 4:13 (NKJV)",
                "For the promise that he would be the heir of the world was not to Abraham or to his seed through the law, but through the righteousness of faith.",
            ),
        ),
        section(
            "3️⃣ Sins Are Forgiven",
            para("Blessed — lawless deeds forgiven, sins covered."),
            verse(
                "Romans 4:7 (NKJV)",
                "Blessed are those whose lawless deeds are forgiven, And whose sins are covered;",
            ),
        ),
        section(
            "4️⃣ Shall Not Impute Sin",
            verse(
                "Romans 4:8 (NKJV)",
                "Blessed is the man to whom the LORD shall not impute sin.",
            ),
            verse(
                "Romans 4:22-24 (KJV)",
                "And therefore it was imputed to him for righteousness. Now it was not written for his sake alone, that it was imputed to him; But for us also, to whom it shall be imputed, if we believe on him that raised up Jesus our Lord from the dead;",
            ),
        ),
        section(
            "📋 Part 2 — Transformation at Age 99",
            para(
                "At <strong>99</strong>, God appeared as <em>Almighty God</em> (Genesis 17:1). "
                "Three changes ended 25 years of delay — <mark><strong>instantly</strong></mark>."
            ),
        ),
        section_h3(
            "Characteristics of Abraham at age 99",
            table(
                ["#", "Change", "Summary"],
                [
                    ["1", "<strong>Hope</strong>", "Spoke God's word · new name Abram → Abraham"],
                    ["2", "<strong>Faith</strong>", "100% Christ on cross · <em>Hey (ה)</em> = grace"],
                    ["3", "<strong>El Shaddai</strong>", "All-Powerful, All-Sufficient — not by my might"],
                ],
            ),
            ul([
                "Weak, almost ready to die — yet blessed <em>instantly</em> at 99",
                "🟢 My power + 🔴 His power = delay · <strong>Spirit alone</strong> = instant blessing",
            ]),
        ),
        section(
            "1. Hope",
            para(
                "<strong>Hope</strong> = proclaim my life <em>will change</em> — speak what God says, "
                "not what circumstances show. God renames — new identity ends delay."
            ),
            table(
                ["Old name", "New name", "Meaning"],
                [
                    ["אַבְרָם Abram", "אַבְרָהָם Abraham", "Father of many nations"],
                    ["שָׂרַי Sarai", "שָׂרָה Sarah", "Princess"],
                ],
            ),
            hebrew_name_changes(),
            ul([
                "Spoke <strong>“father of nations”</strong> with zero children",
                "Hebrew <strong>Hey (ה)</strong> added = grace / undeserved favour",
                "What you speak today shapes tomorrow",
            ]),
            verse(
                "James 3:4-5 (NKJV)",
                "Look also at ships: although they are so large and are driven by fierce winds, they are turned by a very small rudder wherever the pilot desires. Even so the tongue is a little member and boasts great things.",
            ),
            verse(
                "Genesis 17:5 (KJV)",
                "Neither shall thy name any more be called Abram, but thy name shall be Abraham; for a father of many nations have I made thee.",
            ),
            verse(
                "Genesis 17:15 (KJV)",
                "And God said unto Abraham, As for Sarai thy wife, thou shalt not call her name Sarai, but Sarah shall her name be.",
            ),
        ),
        section(
            "2. Faith",
            para(
                "<strong>Faith</strong> = Christ already <u>finished it on the cross</u> — healing, righteousness, victory. "
                "Not blessed by qualifications; by grace through Christ."
            ),
            table(
                ["", "<strong>Hope</strong>", "<strong>Faith</strong>"],
                [
                    ["Speaks about", "My life <em>will change</em>", "Already <u>done on the cross</u>"],
                    ["Proclaims", "What God says about my future", "Victory received in Christ"],
                    ["Who has it?", "Everyone has hope", "Believers in Christ alone"],
                ],
            ),
            table(
                ["Mixture", "Result"],
                [
                    ["🟢 My power + 🔴 His power", "❌ <strong>Delay</strong>"],
                    ["100% His power — Spirit alone", "✅ <mark>Instant blessing</mark>"],
                ],
            ),
            ul([
                "Green = My Power · Red = His Power — <em>no mixture</em>",
                "Lied, brought Lot, Sarah lied — yet faith in Christ counted for righteousness",
            ]),
        ),
        section(
            "3. El Shaddai",
            para(
                "<strong>El Shaddai</strong> — All-Sufficient, All-Powerful God. "
                "Not by my might nor power — by Spirit alone. <u>25 years of delay ended instantly.</u>"
            ),
            verse(
                "Genesis 17:1 (KJV)",
                "And when Abram was ninety years old and nine, the Lord appeared to Abram, and said unto him, I am the Almighty God; walk before me, and be thou perfect.",
            ),
            verse(
                "Zechariah 4:6 (NKJV)",
                "Not by might nor by power, but by My Spirit,' Says the LORD of hosts.",
            ),
            verse(
                "Romans 4:13–14 (KJV)",
                "For the promise, that he should be the heir of the world, was not to Abraham, or to his seed, through the law, but through the righteousness of faith. For if they which are of the law be heirs, faith is made void, and the promise made of none effect:",
            ),
            verse(
                "Galatians 3:29 (KJV)",
                "And if ye be Christ's, then are ye Abraham's seed, and heirs according to the promise.",
            ),
        ),
        sep(),
        h2("🌟 The Gospel in the Stars — Visual Guide"),
        para(
            "Abraham looked at the stars and <em>declared</em> the gospel (not astrology). "
            "Full teaching: <a href=\"https://biblerevelation.org/2026/05/the-gospel-in-the-stars/\">The Gospel in the Stars</a>."
        ),
        GOSPEL_STARS_WHEEL,
        sep(),
        h2("✅ Final Thought"),
        ul([
            "📋 Part 1 (75–85): know gospel · righteousness · sins forgiven",
            "📋 Part 2 (99): <strong>1. Hope</strong> · <strong>2. Faith</strong> · <strong>3. El Shaddai</strong>",
            "Delay → <mark><strong>instant blessing</strong></mark>. Amen.",
        ]),
    ]
    return "\n\n".join(blocks)


if __name__ == "__main__":
    base = "/Users/praison/praisonai-audio-editor/.agent"
    for name, builder in [
        ("biblerevelation-100-fold-blessings.html", build_100_fold),
        ("biblerevelation-how-to-prevent-delay.html", build_delay),
    ]:
        html = builder()
        path = f"{base}/{name}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        h3_count = html.count('level":3')
        print(f"Wrote {path} — h3: {h3_count}, tables: {html.count('wp:table')}, lists: {html.count('wp:list')}")
