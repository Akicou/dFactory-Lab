import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Models from "./pages/Models";
import Datasets from "./pages/Datasets";
import Training from "./pages/Training";
import Export from "./pages/Export";
import Chat from "./pages/Chat";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="models" element={<Models />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="training" element={<Training />} />
        <Route path="export" element={<Export />} />
        <Route path="chat" element={<Chat />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  );
}
